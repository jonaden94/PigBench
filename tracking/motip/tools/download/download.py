# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
import os	
from pathlib import Path
import gdown

BASE_PATH_TRACK = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/P7VQTP/"
BASE_PATH_DET = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/I6UYE9/"

class pigtrack:
    files = [
        ["TSAWHD", "pigtrack0001.zip"],
        ["WB9ZJU", "pigtrack0002.zip"],
        ["4ILP3V", "pigtrack0003.zip"],
        ["L68V3D", "pigtrack0004.zip"],
        ["VWCUPE", "pigtrack0005.zip"],
        ["H5UBL2", "pigtrack0006.zip"],
        ["YLLHTY", "pigtrack0007.zip"],
        ["ZFXZE0", "pigtrack0008.zip"],
        ["YHKNAZ", "pigtrack0009.zip"],
        ["OPYDUQ", "pigtrack0010.zip"],
        ["YMAQCC", "pigtrack0011.zip"],
        ["OFHHHS", "pigtrack0012.zip"],
        ["NJ5TFZ", "pigtrack0013.zip"],
        ["60HWIQ", "pigtrack0014.zip"],
        ["4FZBAM", "pigtrack0015.zip"],
        ["JUSEAV", "pigtrack0016.zip"],
        ["F3NQKQ", "pigtrack0017.zip"],
        ["SQUHLS", "pigtrack0018.zip"],
        ["VY3GHF", "pigtrack0019.zip"],
        ["0SIV94", "pigtrack0020.zip"],
        ["ATCS5X", "pigtrack0021.zip"],
        ["NBYTC5", "pigtrack0022.zip"],
        ["F59N2U", "pigtrack0023.zip"],
        ["MJLSGF", "pigtrack0024.zip"],
        ["G0UM5G", "pigtrack0025.zip"],
        ["IZVWAA", "pigtrack0026.zip"],
        ["6E44EU", "pigtrack0027.zip"],
        ["1PZTUN", "pigtrack0028.zip"],
        ["M22TCF", "pigtrack0029.zip"],
        ["PKEADJ", "pigtrack0030.zip"],
        ["SJHM5R", "pigtrack0031.zip"],
        ["KC7FPC", "pigtrack0032.zip"],
        ["YVNEGA", "pigtrack0033.zip"],
        ["NWOA7S", "pigtrack0034.zip"],
        ["GWPSRK", "pigtrack0035.zip"],
        ["W6XLGG", "pigtrack0036.zip"],
        ["HQCDFE", "pigtrack0037.zip"],
        ["X2VAPW", "pigtrack0038.zip"],
        ["QT61JD", "pigtrack0039.zip"],
        ["MNDKB7", "pigtrack0040.zip"],
        ["JLJUQC", "pigtrack0041.zip"],
        ["YVXAYP", "pigtrack0042.zip"],
        ["FPXOPO", "pigtrack0043.zip"],
        ["SCGSMA", "pigtrack0044.zip"],
        ["4RKGUQ", "pigtrack0045.zip"],
        ["GHGHZX", "pigtrack0046.zip"],
        ["XA8B9T", "pigtrack0047.zip"],
        ["66DZQU", "pigtrack0048.zip"],
        ["1DVUWO", "pigtrack0049.zip"],
        ["KRDO7M", "pigtrack0050.zip"],
        ["HIFGWF", "pigtrack0051.zip"],
        ["UZAWPW", "pigtrack0052.zip"],
        ["PMAMHK", "pigtrack0053.zip"],
        ["VPJQEB", "pigtrack0054.zip"],
        ["SVWSBC", "pigtrack0055.zip"],
        ["ADGB6J", "pigtrack0056.zip"],
        ["8DVRVQ", "pigtrack0057.zip"],
        ["S1VGEW", "pigtrack0058.zip"],
        ["4COXNG", "pigtrack0059.zip"],
        ["HZE4TU", "pigtrack0060.zip"],
        ["N5WQIN", "pigtrack0061.zip"],
        ["BC7HXO", "pigtrack0062.zip"],
        ["PFIZNZ", "pigtrack0063.zip"],
        ["VT0NTI", "pigtrack0064.zip"],
        ["LIXB93", "pigtrack0065.zip"],
        ["QX8N9W", "pigtrack0066.zip"],
        ["OKAMHZ", "pigtrack0067.zip"],
        ["3X6UMG", "pigtrack0068.zip"],
        ["ASLUBF", "pigtrack0069.zip"],
        ["AFZMGD", "pigtrack0070.zip"],
        ["UXNBDB", "pigtrack0071.zip"],
        ["SADTLG", "pigtrack0072.zip"],
        ["NNOIQN", "pigtrack0073.zip"],
        ["PU1G5L", "pigtrack0074.zip"],
        ["GAPYIS", "pigtrack0075.zip"],
        ["BKEENX", "pigtrack0076.zip"],
        ["0WQGHV", "pigtrack0077.zip"],
        ["YSXMM9", "pigtrack0078.zip"],
        ["RMRO5W", "pigtrack0079.zip"],
        ["UOJHLG", "pigtrack0080.zip"],
        ["XUXKRF", "split.txt"]
    ]
    
class pigtrack_videos:
    files = [
        ["HGGTFR", "PigTrackVideos.zip"]
    ]

class pigdetect:
    files = [
        ["YWBQZN", "PigDetect.zip"]
    ]

class motrv2_weights:
    files = [
        ["WPPUCB", "MOTRv2.pth"],
    ]

class motip_weights:
    files = [
        ["XUQFCT", "MOTIP.pth"],
    ]
    
class codino_weights:
    files = [
        ["6YEIHC", "codino_swin.pth"]
    ]
    
class bbox_priors:
    files = [
        ["RNBHTY", "bbox_priors_minconf0.5_rgb.json"]
    ]
    
class detr_pretrained_coco:
    files = [
        ["https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth", 
        "r50_deformable_detr_coco.pth"]
    ]
    
class detr_pretrained_pigs:
    files = [
        ["HZZTJE", "detr_pretrained_pigs.pth"]
    ]

class motrv2_dancetrack:
    files = [
        ["https://drive.google.com/uc?id=1EA4lndu2yQcVgBKR09KfMe5efbf631Th", 
         "motrv2_dancetrack.pth"]
    ]
    
def get_ids(name):
    datasets = {
        "pigtrack": pigtrack,
        "pigtrack_videos": pigtrack_videos,
        "pigdetect": pigdetect,
        "motrv2_weights": motrv2_weights,
        "motip_weights": motip_weights,
        "codino_weights": codino_weights,
        "detr_pretrained_coco": detr_pretrained_coco,
        "detr_pretrained_pigs": detr_pretrained_pigs,
        "motrv2_dancetrack": motrv2_dancetrack,
        "bbox_priors": bbox_priors,
    }
    
    dataset = datasets.get(name)
    if dataset is not None:
        return dataset.files
    else:
        raise NotImplementedError (f"Dataset {name} does not exist.")


def download_data(root, name):
    """ "
    Download data splits specific to a given setting.

    Args:
    root: The root folder where the data will be downloaded
    name: The name of the dataset to download, must be defined in this python file.  """

    print(f"Downloading data for {name} ...")

    # Load and parse metadata csv file
    files = get_ids(name)
    os.makedirs(root, exist_ok=True)

    # Iterate ids and download the files
    for id, save_name in files:
        if name in ['pigtrack', 'motrv2_weights', 'motip_weights', 'bbox_priors', 'pigtrack_videos', 'detr_pretrained_pigs']:
            url = BASE_PATH_TRACK + id
        elif name in ['detr_pretrained_coco', 'motrv2_dancetrack']:
            url = id
        else:
            url = BASE_PATH_DET + id
            
        if name == 'motrv2_dancetrack':
            gdown.download(url, os.path.join(root, save_name), quiet=True)
        else:
            download_url(url, root, save_name)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the PigTrack data",
        epilog="",
    )

    arg_parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset setting to download",
    )

    args = arg_parser.parse_args()

    download_data(args.root, args.name)