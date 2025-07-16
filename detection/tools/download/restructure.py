import zipfile
import os
import shutil

def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    images_dir = 'data/PigDetect/images'
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) == 2931 and not os.path.exists('data/PigDetect/PigDetect.zip'):
        print("PigDetect dataset already restructured.")
    else:
        unzip_file('data/PigDetect/PigDetect.zip', 'data/PigDetect/')
        dev_images_dir = 'data/PigDetect/dev/images'
        test_images_dir = 'data/PigDetect/test/images'
        
        # move all images from dev and test partitions to single images folder as required by coco format
        os.makedirs(images_dir, exist_ok=True)
        for image in os.listdir(dev_images_dir):
            shutil.move(os.path.join(dev_images_dir, image), os.path.join(images_dir, image))
        for image in os.listdir(test_images_dir):
            shutil.move(os.path.join(test_images_dir, image), os.path.join(images_dir, image))
            
        # remove original dev and test folders as well as the zip file
        shutil.rmtree('data/PigDetect/dev')
        shutil.rmtree('data/PigDetect/test')
        os.remove('data/PigDetect/PigDetect.zip')
