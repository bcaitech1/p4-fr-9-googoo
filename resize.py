import os
from glob import glob
from tqdm import tqdm
import cv2

ROOT_DIR = '/opt/ml/input/data/train_dataset/'

def main(to_width, to_height, output_dir, source_dir):
    
    load_dir = ROOT_DIR + source_dir # 원래 이미지 폴더
    save_dir = ROOT_DIR + output_dir + '/' # 만들어진 이후 폴더
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file_path in tqdm(glob(load_dir+'*')):
        
        try:
            
            # 1. 이미지 로드 => cv는 읽는 순서가 다르지만, 그대로 저장할 것이기 때문에 변환은 불필요
            image = cv2.imread(file_path)
            
            # 2. 리사이즈
            image = cv2.resize(image, dsize=(to_width, to_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 3. 이미지 저장
            file_name = file_path.split('/')[-1] # e.g. train_00000.jpg
            cv2.imwrite(save_dir+file_name, image)
            
        except OSError as e:
            
            print(f'error at {f}: {e}')

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source_dir",
        dest="source_dir",
        default="images",
        type=str,
        help="Path of source images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        default="new_images",
        type=str,
        help="Path of output images",
    )
    parser.add_argument(
        "-d",
        "--dest_size",
        dest="dest_size",
        default="128,128",
        type=str,
        help="Size of destination",
    )
    parser = parser.parse_args()
    w, h = map(int, parser.dest_size.split(','))
    print(w, h, parser.output_dir, parser.source_dir)
    main(w, h, output_dir=parser.output_dir, source_dir=parser.source_dir)