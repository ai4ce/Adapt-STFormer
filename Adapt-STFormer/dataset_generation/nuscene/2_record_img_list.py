import os

def get_image_list(directory):
    image_list = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add or remove extensions as needed
                image_list.append(os.path.join(file))
    return image_list

def write_to_file(image_list, filename):
    with open(filename, 'w') as f:
        for image_path in image_list:
            f.write(f"{image_path}\n")
    print(f"Image list has been saved to {filename}")

# Directories to search
directories = [
    'nuscene_small_query',
    'nuscene_small_database',
    'nuscene_large_query',
    'nuscene_large_database'
]

# Process each directory
for directory in directories:
    images = get_image_list(directory)
    
    # Create a filename based on the directory structure, replacing '/' with '_'
    filename = f"{directory.replace('/', '_')}_image_list.txt"
    
    # Use os.path.join to create a path in the current directory
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, filename)
    
    write_to_file(images, full_path)