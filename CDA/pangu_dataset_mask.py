import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
import math
import cv2
from mpmath import mp
from itertools import compress
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def get_ellipse_from_mask( mask ):
    raw_mask = mask

    thresh = 0.9
    raw_mask[raw_mask >= thresh] = 1
    raw_mask[raw_mask < thresh] = 0

    processed_mask = (raw_mask * 255).astype(np.uint8)

    # Extract contours from the processed mask
    contours, _ = cv2.findContours(
        processed_mask,
        cv2.RETR_EXTERNAL,  # Retrieve only external contours
        cv2.CHAIN_APPROX_SIMPLE,  # Simple contour approximation
    )
    # cv2.CHAIN_APPROX_NONE,  # Simple contour approximation
    
    # Overlay contour points on the image
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x,y),(a,b),theta = ellipse
            return x,y,a/2,b/2,theta
        else:
            return 0,0,0,0,0
    return 0,0,0,0,0
            
def plot_masks( img, bboxes, masks, color = ( 1, 0, 0 ), color_channel = 0, plot_box = True, plot_mask = True, plot_ellipse = True):
    bboxes = bboxes.numpy()
    masks = masks.numpy()
    for bbox, mask in zip( bboxes, masks ):
        if plot_box:
            cv2.rectangle(
                img,
                ( int( bbox[0] ), int( bbox[1] ) ),
                ( int( bbox[2] ), int( bbox[3] ) ),
                color,
                4,
            )

        if plot_mask or plot_ellipse:
            # Visualize the raw mask before processing
            raw_mask = mask
            
            thresh = 0.9
            raw_mask[raw_mask >= thresh] = 1
            raw_mask[raw_mask < thresh] = 0
            
            if (plot_mask):
                img[:, :, color_channel] += (raw_mask * 255).astype(np.uint8)

            if (plot_ellipse):
                x,y,a,b,theta = get_ellipse_from_mask(mask)

                # processed_mask = (raw_mask * 255).astype(np.uint8)
                
                # # kernel = np.ones((3, 3), np.uint8)
                # # processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
                # # processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

                # # Extract contours from the processed mask
                # contours, hierarchy = cv2.findContours(
                #     processed_mask,
                #     cv2.RETR_EXTERNAL,  # Retrieve only external contours
                #     cv2.CHAIN_APPROX_SIMPLE,  # Simple contour approximation
                # )
                # # cv2.CHAIN_APPROX_NONE,  # Simple contour approximation
                
                # # Overlay contour points on the image
                # for contour in contours:
                #     # for point in contour:
                #     #     x, y = point[0]  # Extract x and y coordinates
                #     #     cv2.circle(img, (x, y), radius=2, color=(255, 255, 255), thickness=2)  # Draw contour points

                #     if len(contour) >= 5:
                #         ellipse = cv2.fitEllipse(contour)
                #         (x,y),(a,b),theta = ellipse
                #     break
                cv2.ellipse( 
                        img,
                        ( int(x), int(y) ), # Center point
                        ( int(a), int(b) ), # Major and minor axes
                        theta, # Convert angle from radians to degrees
                        0, # Start Angle for drawing
                        360, # End Angle for drawing
                        color,
                        2,
                    )

       
    return img


def plot_ellipses( img, bboxes, ellipse_matrices, color = ( 1, 0, 0 ), plot_boxes = True ):
    for bbox, ellipse_matrix in zip( bboxes, ellipse_matrices ):
        if plot_boxes:
            # Plot bounding box:
            cv2.rectangle(
                img,
                ( int( bbox[0] ), int( bbox[1] ) ),
                ( int( bbox[2] ), int( bbox[3] ) ),
                color,
                4,
            )
        x,y,a,b,theta = ellipse_matrix
        # Plot ellipse:
        cv2.ellipse( 
            img,
            ( int( x), int( y) ), # Center point
            ( int( a), int( b) ), # Major and minor axes
            theta, # Convert angle from radians to degrees
            0, # Start Angle for drawing
            360, # End Angle for drawing
            color,
            2,
        )
    return img

# def plot_ellipse_from_mask( img, masks, color = ( 1, 0, 0 ) ):
#     for  mask in masks:

#         mask = np.array(mask, dtype=np.uint8)
#         mask = mask * 255
#         mask.astype(np.uint8)
        
#         contours, hierarchy = cv2.findContours(
#             mask,
#             cv2.RETR_EXTERNAL,  # Retrieve only external contours
#             cv2.CHAIN_APPROX_SIMPLE  # Simple contour approximation
#         )
        
#         if len(contours) > 0:
#             ellipse = cv2.fitEllipse(contours[0])
#             (x,y),(a,b),theta = ellipse

#         cv2.ellipse( 
#                 img,
#                 ( int(x), int(y) ), # Center point
#                 ( int(a), int(b) ), # Major and minor axes
#                 theta * 180 / math.pi, # Convert angle from radians to degrees
#                 0, # Start Angle for drawing
#                 360, # End Angle for drawing
#                 color,
#                 2,
#             )

#     return img

def load_bounding_boxes( paths ):
    ground_truth = []
    for i, p in enumerate( paths ):
        with open( p ) as f:
            samples = {}
            samples['boxes'] = []
            for line in f.readlines()[1:]:
                line = line.split()
                line = [word.replace( ',', '' ) for word in line]
                
                box = [ int( line[0] ), int( line[1] ), int( line[2] ), int( line[3] ) ]
                box = [ max( min( coord, 1024 ), 0 ) for coord in box ]

                samples['boxes'].append( box )
            ground_truth.append( samples )
    return ground_truth


def load_projected_ellipses_and_ids( path ):
    '''
    Load ellipse projections from file at 'path'
    Read each of the ellipse parameters x, y, a, b, theta
    Convert to tensor
    Then additionally calculate ellipse matrices (in parallel)
    '''
    ids = []
    with open( path ) as f:
        samples = {}
        samples['ellipse_sparse'] = []
        samples['masks'] = []
        crater_ids = []

        for line in f.readlines()[1:]:
            line = line.split()
            line = [ word.replace( ',', '' ) for word in line ]

            ellipse = [
                    float( line[0] ),
                    float( line[1] ),
                    float( line[2] ),
                    float( line[3] ),
                    float( line[4] ),
                ]
            # Change ellipse angle so its within [-pi/2, pi/2]
            if ellipse[4] > math.pi/2:
                ellipse[4] -= math.pi
            elif ellipse[4] < -math.pi/2:
                ellipse[4] += math.pi
            
            samples['ellipse_sparse'].append( ellipse )
            crater_ids.append(line[5])

        samples['ellipse_sparse'] = torch.tensor( samples['ellipse_sparse'], dtype = torch.float32 )
        if len( samples['ellipse_sparse'].shape ) < 2:
            return None
    return samples, np.array(crater_ids)


def load_projected_ellipses( path ):
    '''
    Load ellipse projections from file at 'path'
    Read each of the ellipse parameters x, y, a, b, theta
    Convert to tensor
    Then additionally calculate ellipse matrices (in parallel)
    '''
    with open( path ) as f:
        samples = {}
        samples['ellipse_sparse'] = []
        samples['masks'] = []

        for line in f.readlines()[1:]:
            line = line.split()
            line = [ word.replace( ',', '' ) for word in line ]

            ellipse = [
                    float( line[0] ),
                    float( line[1] ),
                    float( line[2] ),
                    float( line[3] ),
                    float( line[4] ),
                ]
            # Change ellipse angle so its within [-pi/2, pi/2]
            if ellipse[4] > math.pi/2:
                ellipse[4] -= math.pi
            elif ellipse[4] < -math.pi/2:
                ellipse[4] += math.pi
            
            samples['ellipse_sparse'].append( ellipse )

        samples['ellipse_sparse'] = torch.tensor( samples['ellipse_sparse'], dtype = torch.float32 )
        if len( samples['ellipse_sparse'].shape ) < 2:
            return None
    return samples

def compute_mask( image_size, ellipses ):
    '''
    Compute mask from ellipse and bbox
    Masks must be of the shape ( N, 1, H, W )
    N : Number of instances
    H : Image height
    W : Image width
    Returns masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    image_size is tuple ( H, W )
    ellipses is list where each element is [ x, y, a, b, theta ] 
    '''

    masks = []
    for e in ellipses:
        img = np.zeros( image_size )
        
        cv2.ellipse(
            img,
            ( int( e[0].item() ), int( e[1].item() ) ), # Center point
            ( int( e[2].item() ), int( e[3].item() ) ), # Major and minor axes
            e[4].item() * 180 / math.pi, # Convert angle from radians to degrees
            0, # Start Angle for drawing
            360, # End Angle for drawing
            ( 1 ),
            -1, # Mask should be filled
        )
        masks.append( img )
    masks = np.array( masks )
    return torch.tensor( masks, dtype = torch.uint8 )

def load_crater_ids( paths ):
    ground_truth = []
    for i, p in enumerate( paths ):
        with open( p ) as f:
            samples = {}
            samples['crater_id'] = []
            for line in f.readlines()[1:]:
                line = line.split()
                samples['crater_id'].append( line[0] )
            ground_truth.append( samples )
    return np.array( ground_truth )

class CraterDataset( torch.utils.data.Dataset ):
    def __init__( self, catalogue, img_size, root_dir, ldem_name, min_angle, max_angle, angle_inc, max_number_craters = None, require_mask = False, require_ellipse_matrix = False ):
        self.root = root_dir
        self.imgs = []
        self.img_angles = []
        self.img_size = img_size
        self.catalogue = catalogue
        self.max_number_craters = max_number_craters
        self.require_mask = require_mask
        self.require_ellipse_matrix = require_ellipse_matrix

        for angle in range(min_angle, max_angle, angle_inc):
            imgs = list( sorted( 
                os.listdir( self.root + ldem_name+"/" + ldem_name +'_float_60fov_1024_1024_' + str( angle ) + 'deg_off_nadir/ground_truth_images/' )
            ) )
            imgs = [ img for img in imgs if 'ipynb_checkpoints' not in img ]
            self.imgs.extend( [ ldem_name+"/" + ldem_name+'_float_60fov_1024_1024_' + str( angle ) + 'deg_off_nadir/ground_truth_images/' + img for img in imgs ] )
            self.img_angles.extend( [ angle for img in imgs ] )
        
        # Iterate through dataset and check for bad samples
        print( 'Filter bad samples:' )
        mask = Parallel( n_jobs = 16 )( delayed( checkSample )( *[ self, i ] ) for i in range( self.__len__() ) )
        
        self.imgs = list( compress( self.imgs, mask ) )
        self.img_angles = list( compress( self.img_angles, mask ) )

        print( 'Total Images after Filtering', self.__len__() )
    
    def __len__( self ):
        return len( self.imgs )
    
    # Stratified split across each viewing angle
    # Return indices for img paths to test on
    def testSplit( self ):
        # Find how many images from each angle
        trainIndices, testIndices = [], []
        runningTotal = 0
        
        for angle in range( 0, 70, 5 ):
            totalSamples = len( [ img for img in self.imgs if '_' + str( angle ) + 'deg_off_nadir' in img ] )
            trainIndices.extend( list( range( runningTotal, runningTotal + totalSamples * 3 // 4 ) ) ) # 75 - 25 split
            testIndices.extend( list( range( runningTotal + totalSamples * 3 // 4, runningTotal + totalSamples ) ) )
            runningTotal += totalSamples
        return trainIndices, testIndices

    def getTarget( self, idx ):
        # Get bounding boxes
        bboxes = load_bounding_boxes( [ self.root + self.imgs[idx].replace( 'images', 'bounding_boxes' ).replace( 'png', 'txt' ) ] )[0]
        ellipses,ids = load_projected_ellipses_and_ids(self.root + self.imgs[idx].replace( 'images', 'projected_ellipses' ).replace( 'png', 'txt' )) #(sofia)
        # ellipses = load_projected_ellipses( self.root + self.imgs[idx].replace( 'images', 'projected_ellipses' ).replace( 'png', 'txt' ) )
        # ids = load_crater_ids( [ self.root + self.imgs[idx].replace( 'images', 'crater_ids' ).replace( '.png', '.txt' ) ] )[0]

        bboxes = torch.as_tensor( np.array( bboxes['boxes'] ), dtype = torch.float32 )
        labels = torch.ones( len( bboxes ), dtype = torch.int64 )
        
        if len( bboxes.shape ) < 2:
            return None

        # Filter as necessary
        mask = np.array( np.ones( len( bboxes ) ), dtype = bool )
        if np.sum( mask ) > 0: mask = np.logical_and( self.sizeFilter( bboxes, minArea = 25 ), mask )
        ## if np.sum( mask ) > 0: mask = np.logical_and( self.depthFilter( ids, minDepth = 0.2 ), mask )
        # Sofia added:
        # if np.sum( mask ) > 0: mask = np.logical_and( self.ellipticityFilter1( ellipses['ellipse_sparse']), mask )
        if np.sum( mask ) > 0: mask = np.logical_and( self.ellipticityFilter( ellipses['ellipse_sparse']), mask )
        if np.sum( mask ) > 0: mask = np.logical_and( self.christiansFilter( ids, minRimCompleteness = 0.6, minCraterDiameter = 4000, maxCraterDiameter = 125000, maxEllipticity = 1.2), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.christiansFilter( ids, minRimCompleteness = 0.9, minCraterDiameter = 4000, maxCraterDiameter = 125000, maxEllipticity = 1.1), mask )
        if np.sum( mask ) > 0: mask = np.logical_and( self.deepCraterFilter( ids ), mask)
        if self.max_number_craters:
            if np.sum( mask ) > self.max_number_craters: mask = np.logical_and( self.limitNumberOfCraters(ellipses['ellipse_sparse'], mask), mask)

        
        
        # max_num_craters = 5
        # current_num_craters = 0
        # for i in range(len(mask)):
        #     current_num_craters = mask[i] + current_num_craters
        #     if (current_num_craters >= max_num_craters):
        #         mask[i] = 0
            
        
        target = {}
        target['boxes'] = bboxes[mask]
        target['ellipse_sparse'] = ellipses['ellipse_sparse'][mask]
        target['labels'] = labels[mask]
        target['masks'] = compute_mask( self.img_size, target['ellipse_sparse'] )
        target['crater_id'] = ids[mask]
        target['image_id'] = self.root + self.imgs[idx]
        # target['depths'] = depths
        target['view_angle'] = torch.tensor( [ self.img_angles[idx] ] )
        # target['area'] = ( bboxes[:,3] - bboxes[:,1] ) * ( bboxes[:,2] - bboxes[:,0] )
        
        return target
        
    def __getitem__( self, idx ):
        # Load Image
        img = Image.open( self.root + self.imgs[idx] ).convert('L')
        transform = transforms.ToTensor()
        img = transform( img )
        
        target = self.getTarget( idx )
        return img, target
    
    def deepCraterFilter(self, crater_ids):
        # Store only the craters that have rim height greater than the rim threshold.
        rim_threshold = 0.09 #km

        rimMedian = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['rimMedian']
        fullMedian = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['fullMedian']
        craterSemiMinor = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_ELLI_MINOR_IMG']

        rimDiff = ((rimMedian - fullMedian)/craterSemiMinor)*1000

        mask1 = np.array(rimDiff > 0.04, dtype=bool)
        mask1 = np.logical_and(np.array(craterSemiMinor > 5000), mask1)
        

        mask2 = np.array(rimDiff > rim_threshold, dtype=bool)

        return np.logical_or(mask1, mask2)

    
    def christiansFilter(self, crater_ids, minRimCompleteness = 0.9, minCraterDiameter = 4000, maxCraterDiameter = 125000, maxEllipticity = 1.1):
        # Filter craters within Christian's parameters.
        rimCompleteness = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['ARC_IMG']
        craterDiameters = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_CIRC_IMG']*1000
        craterSemiMajor = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_ELLI_MAJOR_IMG']*1000
        craterSemiMinor = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_ELLI_MINOR_IMG']*1000

        mask_rim_completeness = np.array(rimCompleteness > minRimCompleteness, dtype=bool)
        mask_min_diameter = np.array(craterDiameters >= minCraterDiameter, dtype=bool)
        mask_max_diameter = np.array(craterDiameters <= maxCraterDiameter, dtype=bool)
        mask_crater_ellipticiy = np.array(craterSemiMajor/craterSemiMinor <= maxEllipticity)
        mask = np.logical_and( mask_rim_completeness, mask_min_diameter )
        mask = np.logical_and( mask_max_diameter, mask )
        return np.logical_and( mask_crater_ellipticiy, mask )
    
    def limitNumberOfCraters(self, ellipses, mask):
        b_masked = ellipses[:,3][mask]
        # Get element of 100th largest crater if the array was sorted (smalles to largest projected ellipse semi minor ax)
        thresholdSemiMinor = np.partition(b_masked,-self.max_number_craters)[-self.max_number_craters]
        return np.array( ellipses[:,3] >= thresholdSemiMinor, dtype=bool)


    def ellipticityFilter(self, ellipses):
        b = ellipses[:,3]
        min_minor = np.array( b >= 10, dtype=bool)

        return ( min_minor)

    # From pose estimation
    def ellipticityFilter1(self, ellipses):
        a = ellipses[:,2]
        b = ellipses[:,3]
        min_minor = np.array( b >= 10, dtype=bool)

        min_ellipticity_minor = np.array( b >= 5, dtype=bool)
        min_ellipticity = np.array( b/a >= 0.75, dtype=bool)
        min_ellipticity = np.logical_and( min_ellipticity_minor, min_ellipticity)

        return np.logical_or( min_minor, min_ellipticity)
    
    def sizeFilter( self, bboxes, minArea = 25, maxArea = 256 * 256 ):
        areas = ( bboxes[:,3] - bboxes[:,1] ) * ( bboxes[:,2] - bboxes[:,0] )
        min_mask = np.array( areas >= minArea, dtype = bool )
        max_mask = np.array( areas <= maxArea, dtype = bool )
        return np.logical_and( min_mask, max_mask )
    
    def depthFilter( self, crater_ids, minDepth = 0.2 ):
        depths = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['medianDifference']
        return np.array( depths >= minDepth, dtype = np.bool )
    
def checkSample( dataset, i ):
    _, targets = dataset.__getitem__( i )
    if len(targets['boxes']) == 0:
        return False
    if targets is None:
        return False
    return True

def evaluate_ellipse( a, b ):
    '''
    Evaluate error between two ellipses based on:
    - KL Divergence
    - Gaussian Angle
    - Intersection over Union
    - Absolute error in ellipse parameters
    
    Arguments:
    'a' and 'b' are lists such that:
    [ x centre, y centre, semimajor axis, semiminor axis, angle (radians) ]
    '''
    
    error = {}
    error['x_error'] = abs( a[0] - b[0] )
    error['y_error'] = abs( a[1] - b[1] )
    error['a_error'] = abs( a[2] - b[2] )
    error['b_error'] = abs( a[3] - b[3] )
    error['theta_error'] = abs( a[4] - b[4] )
    error['absolute_error'] = np.sum( np.abs( np.array( a ) - np.array( b ) ) )
    
    # Convert sparse ellipse params into conic matrices
    # a_m = ellipse_to_conic_matrix( *wrap_ellipse( [ a[2], a[3], a[0], a[1], a[4], ] ) ).float()
    # b_m = ellipse_to_conic_matrix( *wrap_ellipse( [ b[2], b[3], b[0], b[1], b[4], ] ) ).float()
    
    # error['gaussian_angle'] = gaussian_angle_distance( a_m, b_m ).item()
    # error['kl_divergence'] = norm_mv_kullback_leibler_divergence( a_m.unsqueeze( 0 ), b_m.unsqueeze( 0 ) ).item()
    
    # Intersection over union!
    img_shape = ( 1024, 1024, 3 )
    img1 = np.zeros( img_shape )
    img2 = np.zeros( img_shape )
    
    # Draw predicted ellipse in Red channel (filled)
    cv2.ellipse(
        img1,
        ( int( a[0] ), int( a[1] ) ), # Center point
        ( int( a[2] ), int( a[3] ) ), # Semiminor and Semimajor axes
        float( a[4] * 180 / math.pi ), # Angle (convert from radians to degrees)
        0, # Start Angle for drawing
        360, # End Angle for drawing
        ( 1, 0, 0 ),
        -1,
    )
    
    cv2.ellipse(
        img2,
        ( int( b[0] ), int( b[1] ) ), # Center point
        ( int( b[2] ), int( b[3] ) ), # Semiminor and Semimajor axes
        float( b[4] * 180 / math.pi ), # Angle (convert from radians to degrees)
        0, # Start Angle for drawing
        360, # End Angle for drawing
        ( 1, 0, 0 ),
        -1,
    )
    
    intersection = np.logical_and( img1[:,:,0], img2[:,:,0] )
    union = np.logical_or( img1[:,:,0], img2[:,:,0] )
    error['IoU'] = np.sum( intersection ) / np.sum( union )
    return error



class CraterPredictionDataset( torch.utils.data.Dataset ):
    def __init__( self, catalogue, img_size, root_dir, ldem_name, min_angle, max_angle, angle_inc, max_number_craters = None, require_mask = False, require_ellipse_matrix = False ):
        self.root = root_dir
        self.imgs = []
        self.img_size = img_size
        self.catalogue = catalogue
        self.max_number_craters = max_number_craters
        self.require_mask = require_mask
        self.require_ellipse_matrix = require_ellipse_matrix

        imgs = list( sorted( 
            os.listdir( self.root + ldem_name+"/ground_truth_images/" )
        ) )

        imgs = [ img for img in imgs if 'ipynb_checkpoints' not in img ]
        self.imgs.extend( [ ldem_name+"/ground_truth_images/" + img for img in imgs ] )
        
        # Iterate through dataset and check for bad samples
        print( 'Filter bad samples:' )
        mask = Parallel( n_jobs = 16 )( delayed( checkSample )( *[ self, i ] ) for i in range( self.__len__() ) )
        
        self.imgs = list( compress( self.imgs, mask ) )
        # self.img_angles = list( compress( self.img_angles, mask ) )

        print( 'Total Images after Filtering', self.__len__() )
    
    def __len__( self ):
        return len( self.imgs )
    

    def getTarget( self, idx ):
        # Get bounding boxes
        bboxes = load_bounding_boxes( [ self.root + self.imgs[idx].replace( 'images', 'bounding_boxes' ).replace( 'png', 'txt' ) ] )[0]
        ellipses,ids = load_projected_ellipses_and_ids(self.root + self.imgs[idx].replace( 'images', 'projected_ellipses' ).replace( 'png', 'txt' )) #(sofia)
        # ellipses = load_projected_ellipses( self.root + self.imgs[idx].replace( 'images', 'projected_ellipses' ).replace( 'png', 'txt' ) )
        # ids = load_crater_ids( [ self.root + self.imgs[idx].replace( 'images', 'crater_ids' ).replace( '.png', '.txt' ) ] )[0]

        bboxes = torch.as_tensor( np.array( bboxes['boxes'] ), dtype = torch.float32 )
        labels = torch.ones( len( bboxes ), dtype = torch.int64 )
        
        if len( bboxes.shape ) < 2:
            return None

        # Filter as necessary
        mask = np.array( np.ones( len( bboxes ) ), dtype = bool )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.sizeFilter( bboxes, minArea = 25 ), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.depthFilter( ids, minDepth = 0.2 ), mask )
        # Sofia added:
        # if np.sum( mask ) > 0: mask = np.logical_and( self.ellipticityFilter1( ellipses['ellipse_sparse']), mask )
        if np.sum( mask ) > 0: mask = np.logical_and( self.ellipticityFilter( ellipses['ellipse_sparse']), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.christiansFilter( ids, minRimCompleteness = 0.6, minCraterDiameter = 4000, maxCraterDiameter = 125000, maxEllipticity = 1.2), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.christiansFilter( ids, minRimCompleteness = 0.9, minCraterDiameter = 4000, maxCraterDiameter = 125000, maxEllipticity = 1.1), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.deepCraterFilter( ids ), mask)
        # if self.max_number_craters:
        #     if np.sum( mask ) > self.max_number_craters: mask = np.logical_and( self.limitNumberOfCraters(ellipses['ellipse_sparse'], mask), mask)

        
        
        # max_num_craters = 5
        # current_num_craters = 0
        # for i in range(len(mask)):
        #     current_num_craters = mask[i] + current_num_craters
        #     if (current_num_craters >= max_num_craters):
        #         mask[i] = 0
            
        
        target = {}
        target['boxes'] = bboxes[mask]
        target['ellipse_sparse'] = ellipses['ellipse_sparse'][mask]
        target['labels'] = labels[mask]
        target['masks'] = compute_mask( self.img_size, target['ellipse_sparse'] )
        target['crater_id'] = ids[mask]
        target['image_id'] = self.root + self.imgs[idx]
        # target['depths'] = depths
        # target['view_angle'] = torch.tensor( [ self.img_angles[idx] ] )
        # target['area'] = ( bboxes[:,3] - bboxes[:,1] ) * ( bboxes[:,2] - bboxes[:,0] )
        
        return target
        
    def __getitem__( self, idx ):
        # Load Image
        img = Image.open( self.root + self.imgs[idx] ).convert('L')
        transform = transforms.ToTensor()
        img = transform( img )
        
        target = self.getTarget( idx )
        return img, target
    
    
    def deepCraterFilter(self, crater_ids):
        # Store only the craters that have rim height greater than the rim threshold.
        rim_threshold = 0.09 #km

        rimMedian = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['rimMedian']
        fullMedian = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['fullMedian']
        craterSemiMinor = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_ELLI_MINOR_IMG']

        rimDiff = ((rimMedian - fullMedian)/craterSemiMinor)*1000

        mask1 = np.array(rimDiff > 0.04, dtype=bool)
        mask1 = np.logical_and(np.array(craterSemiMinor > 5000), mask1)
        

        mask2 = np.array(rimDiff > rim_threshold, dtype=bool)

        return np.logical_or(mask1, mask2)

    
    def christiansFilter(self, crater_ids, minRimCompleteness = 0.9, minCraterDiameter = 4000, maxCraterDiameter = 125000, maxEllipticity = 1.1):
        # Filter craters within Christian's parameters.
        rimCompleteness = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['ARC_IMG']
        craterDiameters = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_CIRC_IMG']*1000
        craterSemiMajor = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_ELLI_MAJOR_IMG']*1000
        craterSemiMinor = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['DIAM_ELLI_MINOR_IMG']*1000

        mask_rim_completeness = np.array(rimCompleteness > minRimCompleteness, dtype=bool)
        mask_min_diameter = np.array(craterDiameters >= minCraterDiameter, dtype=bool)
        mask_max_diameter = np.array(craterDiameters <= maxCraterDiameter, dtype=bool)
        mask_crater_ellipticiy = np.array(craterSemiMajor/craterSemiMinor <= maxEllipticity)
        mask = np.logical_and( mask_rim_completeness, mask_min_diameter )
        mask = np.logical_and( mask_max_diameter, mask )
        return np.logical_and( mask_crater_ellipticiy, mask )
    
    def limitNumberOfCraters(self, ellipses, mask):
        b_masked = ellipses[:,3][mask]
        # Get element of 100th largest crater if the array was sorted (smalles to largest projected ellipse semi minor ax)
        thresholdSemiMinor = np.partition(b_masked,-self.max_number_craters)[-self.max_number_craters]
        return np.array( ellipses[:,3] >= thresholdSemiMinor, dtype=bool)


    def ellipticityFilter(self, ellipses):
        b = ellipses[:,3]
        min_minor = np.array( b >= 10, dtype=bool)

        return ( min_minor)

    # From pose estimation
    # def ellipticityFilter(self, ellipses):
    #     a = ellipses[:,2]
    #     b = ellipses[:,3]
    #     min_minor = np.array( b >= 10, dtype=bool)

    #     min_ellipticity_minor = np.array( b >= 5, dtype=bool)
    #     min_ellipticity = np.array( b/a >= 0.75, dtype=bool)
    #     min_ellipticity = np.logical_and( min_ellipticity_minor, min_ellipticity)

        # return np.logical_or( min_minor, min_ellipticity)
    
    def sizeFilter( self, bboxes, minArea = 25, maxArea = 256 * 256 ):
        areas = ( bboxes[:,3] - bboxes[:,1] ) * ( bboxes[:,2] - bboxes[:,0] )
        min_mask = np.array( areas >= minArea, dtype = bool )
        max_mask = np.array( areas <= maxArea, dtype = bool )
        return np.logical_and( min_mask, max_mask )
    
    def depthFilter( self, crater_ids, minDepth = 0.2 ):
        depths = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['medianDifference']
        return np.array( depths >= minDepth, dtype = np.bool )