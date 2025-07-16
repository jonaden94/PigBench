import os
import zipfile

# make sure script works from wherever it is run
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def remove_unwanted_folders(directory, keep_folders):
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if item not in keep_folders:
            os.remove(path)

if __name__ == '__main__':
    unzip_path = '../../data/datasets/PigTrackVideos'
    zip_file = os.path.join(unzip_path, 'PigTrackVideos.zip')

    unzip_file(zip_file, unzip_path)
    os.remove(zip_file)

    # Remove all folders except these
    keep = {'pigtrack0004.mp4', 'pigtrack0010.mp4'}
    remove_unwanted_folders(unzip_path, keep)
