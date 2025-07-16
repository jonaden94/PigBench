import os
import zipfile
import sys

if os.path.exists("data/datasets/PigTrack/test_seqmap_2videos_only.txt"):
    print("PigTrack dataset already restructured.")
    sys.exit(0)

# Base folder containing the ZIPs and split.txt
base_dir = "data/datasets/PigTrack"
split_file = os.path.join(base_dir, "split.txt")

# Read split.txt and create a mapping of filename (without .zip) to split
split_map = {}
with open(split_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        assert len(parts) == 2, f"Invalid line in split.txt: {line.strip()}"
        name, split = parts
        split_map[name] = split

# Process each zip file
for file_name in os.listdir(base_dir):
    if file_name == "split.txt" or file_name in ['train', 'val', 'test'] or file_name in ['train_seqmap.txt', 'val_seqmap.txt', 'test_seqmap.txt']:
        continue
    assert file_name.endswith(".zip"), f"File {file_name} is not a zip file"
    base_name = file_name[:-4]  # Remove .zip extension
    assert base_name in split_map, f"File {base_name} not found in split.txt"

    # Determine split folder
    split = split_map[base_name]
    output_dir = os.path.join(base_dir, split, base_name)

    # Create split folder if not exist
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(base_dir, file_name)

    # Extract zip contents
    print(f"Extracting {file_name} -> data/datasets/PigTrack/{split}/")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Delete the zip file
    os.remove(zip_path)
    print(f"Deleted {file_name}")

# --- Generate seqmap files ---
splits = {"train": [], "val": [], "test": []}

# Populate splits
for name, split in split_map.items():
    if split in splits:
        splits[split].append(name)

# Write seqmap files
for split_name, names in splits.items():
    out_path = os.path.join(base_dir, f"{split_name}_seqmap.txt")
    with open(out_path, "w") as f:
        f.write("name\n")
        sorted_names = sorted(names)  # Optional: sort alphabetically
        for i, name in enumerate(sorted_names):
            if i < len(sorted_names) - 1:
                f.write(f"{name}\n")
            else:
                f.write(f"{name}")  # No newline after last entry
                
# write seqmap file for 2 videos only to illustrate evaluation on two videos
out_path = os.path.join(base_dir, "test_seqmap_2videos_only.txt")
with open(out_path, "w") as f:
    f.write("name\n")
    f.write("pigtrack0004\n")
    f.write("pigtrack0010")
