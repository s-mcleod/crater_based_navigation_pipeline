import re
import matplotlib.pyplot as plt

def process_crater_data(file_path):
    # Initialize counters and data containers
    matching_rate_non_zero_count = 0
    matching_rate_zero_count = 0
    lines_with_non_zero_matching_and_false = []
    matching_rates = []
    false_matching_rates = []
    false_negative_rates = []
    total_craters = []
    total_lines = 0

    # Read and process the file
    with open(file_path, 'r') as file:
        for line in file:
            total_lines += 1
            # Extract relevant data using regex
            match = re.search(r"Testing ID: \d+ \| Matched IDs: (.*?) \| Matching Rate: ([0-9.]+) \| False Matching Rate: ([0-9.]+) \| False Negative Rate: ([0-9.]+)", line)
            if match:
                matched_ids = match.group(1).split(', ')
                matching_rate = float(match.group(2))
                false_matching_rate = float(match.group(3))
                false_negative_rate = float(match.group(4))

                # Count total craters (non-None entries)
                crater_count = len(matched_ids)
                total_craters.append(crater_count)
                matching_rates.append(matching_rate)
                false_matching_rates.append(false_matching_rate)
                false_negative_rates.append(false_negative_rate)

                # Update counters
                if matching_rate != 0:
                    matching_rate_non_zero_count += 1

                    # Check for non-zero false matching rate
                    if false_matching_rate != 0:
                        lines_with_non_zero_matching_and_false.append((crater_count, matching_rate, false_matching_rate))
                else:
                    matching_rate_zero_count += 1

    # Plot matching rate vs number of craters
    plt.figure(figsize=(10, 6))
    plt.scatter(total_craters, matching_rates, alpha=0.7, label='Matching Rate')
    plt.title('Number of Craters vs Matching Rate')
    plt.xlabel('Total Number of Craters')
    plt.ylabel('Matching Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('craters_vs_matching_rate.png')
    plt.close()

    # Plot false matching rate vs number of craters
    plt.figure(figsize=(10, 6))
    plt.scatter(total_craters, false_matching_rates, alpha=0.7, label='False Matching Rate', color='orange')
    plt.title('Number of Craters vs False Matching Rate')
    plt.xlabel('Total Number of Craters')
    plt.ylabel('False Matching Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('craters_vs_false_matching_rate.png')
    plt.close()

    # Plot matching rate vs false matching rate
    plt.figure(figsize=(10, 6))
    plt.scatter(matching_rates, false_matching_rates, alpha=0.7, label='Matching Rate vs False Matching Rate', color='green')
    plt.title('Matching Rate vs False Matching Rate')
    plt.xlabel('Matching Rate')
    plt.ylabel('False Matching Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('matching_rate_vs_false_matching_rate.png')
    plt.close()

    # Calculate percentages
    matching_rate_non_zero_percentage = (matching_rate_non_zero_count / total_lines) * 100
    false_matching_rate_non_zero_count = sum(1 for rate in false_matching_rates if rate != 0)
    false_matching_rate_non_zero_percentage = (false_matching_rate_non_zero_count / total_lines) * 100
    false_negative_rate_equal_one_count = sum(1 for rate in false_negative_rates if rate == 1)
    false_negative_rate_equal_one_percentage = (false_negative_rate_equal_one_count / total_lines) * 100

    # Return results
    return {
        "matching_rate_non_zero_count": matching_rate_non_zero_count,
        "matching_rate_zero_count": matching_rate_zero_count,
        "matching_rate_non_zero_percentage": matching_rate_non_zero_percentage,
        "false_matching_rate_non_zero_percentage": false_matching_rate_non_zero_percentage,
        "false_negative_rate_equal_one_percentage": false_negative_rate_equal_one_percentage,
        "histogram_file_1": 'craters_vs_matching_rate.png',
        "histogram_file_2": 'craters_vs_false_matching_rate.png',
        "histogram_file_3": 'matching_rate_vs_false_matching_rate.png'
    }

# Example usage
file_path = 'result_temp.txt'  # Replace with the path to your file
results = process_crater_data(file_path)

print(f"Lines with non-zero matching rate: {results['matching_rate_non_zero_count']} ({results['matching_rate_non_zero_percentage']:.2f}%)")
print(f"Lines with zero matching rate: {results['matching_rate_zero_count']} ({100 - results['matching_rate_non_zero_percentage']:.2f}%)")
print(f"Percentage of lines with non-zero false matching rate: {results['false_matching_rate_non_zero_percentage']:.2f}%")
print(f"Percentage of lines with False Negative Rate equal to 1: {results['false_negative_rate_equal_one_percentage']:.2f}%")
print(f"Graphs saved as {results['histogram_file_1']}, {results['histogram_file_2']}, and {results['histogram_file_3']}")