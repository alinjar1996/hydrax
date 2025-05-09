import os
import re

# Define the file path
file_path = f"{os.path.dirname(__file__)}/ur5e_2.xml"
output_path = f"{os.path.dirname(__file__)}/ur5e_1_updated.xml"

# Read the content of the XML file
with open(file_path, "r") as file:
    content = file.read()

# Replace all occurrences of "_2" at the end of names, class, or other attributes with "_1"
updated_content = re.sub(r"(\b\w+)_2\b", r"\1_1", content)

# Save the updated content to a new file
with open(output_path, "w") as file:
    file.write(updated_content)

print(f"Updated XML saved to {output_path}")
