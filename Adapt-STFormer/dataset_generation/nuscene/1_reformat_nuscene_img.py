import os
import re

# Define the directory containing the images
directory = 'nuscene_test_database'

# Define the regex pattern to extract the timestamp
pattern = re.compile(r'__(\d+)\.jpg$')

# Open a log file to write the rename actions
with open(os.path.join(os.getcwd(), f'{directory}_rename_log.txt'), 'w') as log_file:
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Match the pattern to extract the timestamp
        match = pattern.search(filename)
        if match:
            # Get the timestamp
            timestamp = match.group(1)
            # Define the new filename
            new_filename = f'{timestamp}.jpg'
            # Construct full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            # Write the log
            log_file.write(f'Renamed: {old_file} -> {new_file}\n')
