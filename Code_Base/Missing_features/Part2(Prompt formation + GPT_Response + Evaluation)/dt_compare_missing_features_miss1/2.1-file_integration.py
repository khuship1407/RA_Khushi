import json
import string
import re
import pandas as pd

# Function to parse the GPT response
def parse_response(response):
    """Parse the response to extract the traffic type and include all subsequent content, excluding only the first line."""
    # Define the list of valid traffic types
    valid_traffic_types = ['BenignTraffic', 'DDoS', 'Brute_Force', 'Spoofing', 'DoS', 'Recon', 'Web-Based', 'Mirai']

    # Primary extraction: handle plain text, markdown bold (**X**), missing blank lines
    m = re.search(r'Most likely type of traffic:\s*\*{0,2}\s*([\w-]+)', response)
    candidate = m.group(1).split('\n')[0].strip() if m else None

    # Fallback: scan full response for any valid label
    if candidate not in valid_traffic_types:
        fb = re.search(
            r'\b(BenignTraffic|DDoS|Brute_Force|Spoofing|DoS|Recon|Web-Based|Mirai)\b', response
        )
        candidate = fb.group(1) if fb else None

    if candidate not in valid_traffic_types:
        print(f"Invalid traffic type encountered: '{candidate}'.")
        return None

    traffic_type = candidate
    first_newline = response.find('\n')
    chosen_path_and_rationale = response[first_newline:].strip() if first_newline != -1 else ''

    return {
        'traffic_type': traffic_type,
        'chosen_path_and_rationale': chosen_path_and_rationale
    }

# Load the initial data
with open('dt_compare_missing_features_miss1/input/matching_integrated_data.json', 'r') as file:
    original_data = json.load(file)

# Load the responses from GPT
with open('dt_compare_missing_features_miss1/output/analysis_input_results.jsonl', 'r') as file:
    responses = [json.loads(line) for line in file.readlines()]

# Check that the number of responses matches the number of entries
assert len(responses) == len(original_data), "Mismatch in number of entries and responses"

correct_predictions = 0

# Process each response and update the original data
for entry, response in zip(original_data, responses):
    parsed_response = parse_response(response)
    choice = parsed_response['traffic_type']
    reason = parsed_response['chosen_path_and_rationale']
    
    # Convert "first" or "second" to the corresponding prediction
    gpt_choice = choice

    if gpt_choice == entry[1]:
        correct_predictions += 1
        
    # Update the entry
    entry.append(gpt_choice)
    entry.append(reason)
    
# Save the updated data to a new file
with open('dt_compare_missing_features_miss1/output/analysis_input_results_without_prob.json', 'w') as file:
    json.dump(original_data, file, indent=4)
    
accuracy = correct_predictions / len(original_data)
print(f"Number of correct choice(out of {str(len(original_data))}): " + str(correct_predictions))
print("Accuracy: " + str(accuracy))

print("Updated data with GPT responses saved successfully.")