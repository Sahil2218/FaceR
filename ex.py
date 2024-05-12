import os

# Get the current directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Get the absolute path of a file in the current directory
file_name = "index2.html"
file_path = os.path.join(current_dir, file_name)
print("File Path:", file_path)
