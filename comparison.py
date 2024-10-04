import pandas as pd
from LlamaGeneration import generate
import re

# Generate the prompt for LLaMA
def generate_llama_prompt(control1, control2):
    return f"Compare the cybersecurity control from Framework 1: '{control1}' with the control from Framework 2: '{control2}'. Respond only with 'Score: X' where X is 1 for full match, 5 for partial match, and 0 for no match."

# Extract the score from LLaMA's response
def extract_score(response):
    match = re.search(r'Score:\s*(\d+)', response)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No score found in LLaMA response: {response}")

# Store results for each control comparison
def store_results(results_list, s_no, control1, full_matches, partial_matches):
    results_list.append({
        's.no.': s_no,
        'Control from Framework 1': control1,
        'Controls from Framework 2 (Full Match)': ', '.join(full_matches),
        'Controls from Framework 2 (Partial Match)': ', '.join(partial_matches)
    })

# Compare frameworks by control labels and use LLaMA to generate similarity scores
def compare_frameworks(framework1, framework2):
    results_list = []  # List to store the comparison results
    s_no = 1  # Serial number for each comparison
    
    for label in framework1:
        if label in framework2:  # Compare only matching labels
            for control1 in framework1[label]:
                full_matches = []    # List to store full matches
                partial_matches = []  # List to store partial matches
                
                for control2 in framework2[label]:
                    prompt = generate_llama_prompt(control1, control2)
                    response = generate(prompt)  # Get LLaMA's response
                    score = extract_score(response)  # Extract numeric score from response

                    # Categorize based on score
                    if score == 1:  # Full match
                        full_matches.append(control2)
                    elif score >= 3:  # Partial match
                        partial_matches.append(control2)

                # Store the results for the current control1
                store_results(results_list, s_no, control1, full_matches, partial_matches)
                s_no += 1
    
    # Convert the results list to a pandas DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df

# Example usage with simple frameworks
framework1 = {'Access Control': ['Ensure password complexity', 'Restrict privileged accounts'], 
              'Incident Response': ['Develop incident response plan']}
framework2 = {'Access Control': ['Passwords must be complex', 'Limit admin accounts'], 
              'Incident Response': ['Incident response process should be in place']}

# Run comparison
results_df = compare_frameworks(framework1, framework2)
print(results_df)
