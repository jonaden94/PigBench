import os
from PIL import Image
import glob
import zipfile

# make sure script works from wherever it is run
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def restructure_yolo_annotations(image_dir, annotations_input_dir, annotations_output_dir):
    if not os.path.exists(annotations_output_dir):
        os.makedirs(annotations_output_dir)
    
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    annotation_files = glob.glob(os.path.join(annotations_input_dir, '*.txt'))
    annotation_dict = {os.path.basename(f).replace('.txt', ''): f for f in annotation_files}
    detection_id = 0
    
    for image_path in image_files:
        image_name = os.path.basename(image_path).replace('.jpg', '')
        annotation_path = annotation_dict.get(image_name, None)
        
        if not annotation_path:
            continue
        
        with open(annotation_path, 'r') as file:
            annotations = file.readlines()
        
        image = Image.open(image_path)
        img_width, img_height = image.size
        output_lines = []
        for annotation in annotations:
            parts = annotation.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            bb_left = int((x_center - width / 2) * img_width)
            bb_top = int((y_center - height / 2) * img_height)
            bb_width = int(width * img_width)
            bb_height = int(height * img_height)
            
            output_lines.append(f"{class_id} {detection_id} {bb_left} {bb_top} {bb_width} {bb_height}\n")
            detection_id += 1
        
        output_path = os.path.join(annotations_output_dir, f"{image_name}.txt")
        with open(output_path, 'w') as output_file:
            output_file.writelines(output_lines)
    
if __name__ == '__main__':
    annotations_output_dir_dev = '../../data/datasets/PigDetect/dev/gts'
    annotations_output_dir_test = '../../data/datasets/PigDetect/test/gts'
    if os.path.exists(annotations_output_dir_dev) and os.path.exists(annotations_output_dir_test) \
        and len(os.listdir(annotations_output_dir_dev)) == 2681 and len(os.listdir(annotations_output_dir_test)) == 250:
            print("PigDetect dataset already restructured.")
    else:
        unzip_file('../../data/datasets/PigDetect/PigDetect.zip', '../../data/datasets/PigDetect')
        os.remove('../../data/datasets/PigDetect/PigDetect.zip')

        splits = ['dev', 'test']
        for split in splits:
            image_dir = f'../../data/datasets/PigDetect/{split}/images'
            annotations_input_dir = f'../../data/datasets/PigDetect/{split}/labels'
            annotations_output_dir = f'../../data/datasets/PigDetect/{split}/gts'
            restructure_yolo_annotations(image_dir, annotations_input_dir, annotations_output_dir)
