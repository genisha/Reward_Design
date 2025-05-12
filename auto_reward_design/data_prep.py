import pandas as pd
import json
from pathlib import Path

# Load the CSV file
csv_path = "data/Qwen0_5_dpo.csv"  # Update this path as needed
df = pd.read_csv(csv_path)

# Prepare the list for JSON output
json_data = []

# Function to escape newlines
def escape_newlines(text):
    return str(text).strip().replace("\n", "\n\n")

# Iterate through each row and construct the JSON structure
for _, row in df.iterrows():
    prompt = escape_newlines(row['Prompt'])
    output_a = escape_newlines(row['OutputA'])
    output_b = escape_newlines(row['OutputB'])
    preferred = row['OutputA_preferred']

    # Skip rows with missing preference
    if pd.isna(preferred):
        continue

    # Assign chosen and rejected based on preference
    if preferred == 1.0:
        chosen = output_a
        rejected = output_b
    else:
        chosen = output_b
        rejected = output_a

    # Append the structured dictionary
    json_data.append({
        "prompt": [{"content": prompt, "role": "user"}],
        "chosen": [{"content": chosen, "role": "assistant"}],
        "rejected": [{"content": rejected, "role": "assistant"}]
    })

# Save to JSON file
output_path = Path("data/Qwen0_5_dpo_converted.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"JSON file saved to: {output_path}")
