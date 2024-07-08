import torch
import sys
import os
import argparse
import pickle
import PIL.Image
import csv
import glob
import subprocess
import pandas as pd





def main():
    out_dir = "./result"
    imgdir = "./out"
    torch.backends.cudnn.benchmark = True


    #print(sys.path)
    os.makedirs(out_dir, exist_ok=True)

    with open(f'{out_dir}/test_imgs.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img_path'])

        # Get image file paths
        img_paths = glob.glob(f'{imgdir}/*')
        print(img_paths)

        # Write image file paths
        for path in img_paths:
            modified_path = f"../{path}"
            writer.writerow([modified_path])


if __name__ == "__main__":
    main()

    

