import os
import json

# Paths
input_dir = r"D:\Data\mpep_finetune"
output_file = r"D:\Data\mpep_finetune\mpep_data.jsonl"

# Placeholder for reserved sections
reserved_placeholder = "This section is reserved and does not contain any additional information."

# Function to process each section and create instruction-response pairs
def process_section(title, section):
    instruction = f"What does MPEP {title} - {section['section_title']} cover?"
    response = " ".join(section['content']).strip()
    # Replace empty response with the reserved placeholder
    if not response:
        response = reserved_placeholder
    return {"instruction": instruction, "response": response}

# Process all JSON files and convert them to JSONL format
with open(output_file, "w", encoding="utf-8") as out_file:
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", "Unknown Title")
                    sections = data.get("sections", [])
                    for section in sections:
                        entry = process_section(title, section)
                        json.dump(entry, out_file)
                        out_file.write("\n")

print(f"Preprocessed data saved to {output_file}")
