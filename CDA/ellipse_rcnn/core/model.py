from typing import List, Tuple, Optional

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import Conv2d
from torch.optim import SGD, Optimizer
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from .roi_heads import EllipseRoIHeads, EllipseRegressor
from ..utils.types import CollatedBatchType


class EllipseRCNN(GeneralizedRCNN, pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        # transform parameters
        backbone_name: str = "resnet50",
        min_size: int = 256,
        max_size: int = 512,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # RPN parameters
        rpn_anchor_generator: Optional[nn.Module] = None,
        rpn_head: Optional[nn.Module] = None,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_score_thresh: float = 0.0,
        # Box parameters
        box_roi_pool: Optional[nn.Module] = None,
        box_head: Optional[nn.Module] = None,
        box_predictor: Optional[nn.Module] = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        bbox_reg_weights: Optional[Tuple[float, float, float, float]] = None,
        # Ellipse regressor
        ellipse_roi_pool: Optional[nn.Module] = None,
        ellipse_head: Optional[nn.Module] = None,
        ellipse_predictor: Optional[nn.Module] = None,
        ellipse_loss_metric: str = "gaussian-angle",
    ):

        backbone = resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=5)

        # Input image is grayscale -> in_channels = 1 instead of 3 (COCO)
        backbone.body.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor " "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            if isinstance(resolution, int):
                box_head = TwoMLPHead(out_channels * resolution**2, representation_size)
            else:
                raise ValueError("resolution should be an int but is {}".format(resolution))

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        if ellipse_roi_pool is None:
            ellipse_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if ellipse_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            if isinstance(resolution, int):
                ellipse_head = TwoMLPHead(out_channels * resolution**2, representation_size)
            else:
                raise ValueError("resolution should be an int but is {}".format(resolution))

        if ellipse_predictor is None:
            representation_size = 1024
            ellipse_predictor = EllipseRegressor(representation_size, num_classes)

        roi_heads = EllipseRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            # Ellipse
            ellipse_roi_pool=ellipse_roi_pool,
            ellipse_head=ellipse_head,
            ellipse_predictor=ellipse_predictor,
            ellipse_loss_metric=ellipse_loss_metric,
        )

        if image_mean is None:
            image_mean = [0.156]
        if image_std is None:
            image_std = [0.272]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_heads, transform)

    def configure_optimizers(self) -> Optimizer:
        return SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

    def training_step(self, batch: CollatedBatchType, batch_idx: int = 0) -> torch.Tensor:
        images, targets = batch
        loss_dict = self(images, targets)
        for name, value in loss_dict.items():
            self.log(name, value, prog_bar=True, logger=True, on_step=True)

        loss = sum(loss_dict.values())
        self.log("total_loss", loss, prog_bar=True, logger=True, on_step=True)

        return loss
