import os

# Directory where you want to create subfolders
parent_directory = './audio/noise_levels/'

# Create 1000 subfolders
for i in range(0, 1000000, 1000):
    subfolder_name = f'{i}'
    subfolder_path = os.path.join(parent_directory, subfolder_name)
    os.makedirs(subfolder_path)

print("Subfolders created successfully.")