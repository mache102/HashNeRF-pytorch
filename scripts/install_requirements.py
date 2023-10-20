import subprocess
from tqdm import tqdm

# Define the path to your requirements.txt file
requirements_file = 'requirements.txt'

# Read the original requirements.txt file
with open(requirements_file, 'r') as file:
    original_requirements = file.readlines()

# Create a list to store the modified requirements
modified_requirements = []

# Iterate through the original requirements and check for version constraints
for line in tqdm(original_requirements):
    # Use a subprocess to try to install the package and check for errors
    try:
        subprocess.check_output(['pip', 'install', line.strip()])
        # If installation is successful, add the line to the modified requirements
        modified_requirements.append(line)
    except subprocess.CalledProcessError:
        # If there's an error (e.g., no version found), skip the package
        print(f"Skipping: {line.strip()}")

# Write the modified requirements back to the requirements.txt file
with open(requirements_file, 'w') as file:
    file.writelines(modified_requirements)
