from PIL import Image
import argparse
import pyocr
import cv2
import pyocr.builders
import pytesseract
from pytesseract import Output
import os
import subprocess

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract/tessdata/'

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--out_path', type=str, default=None)
args = parser.parse_args()

def rotate_and_save_image(img_path, angle, out_path):
    rotate_command = f"convert {img_path} -rotate {angle} -quality 100 {out_path}"
    subprocess.run(rotate_command, shell=True, check=True)
    return 0

def detect_and_rotate_image(img_path, out_path):
    osd_info = pytesseract.image_to_osd(Image.open(img_path), output_type=Output.DICT)
    print(f"OSD Info: {osd_info}")
    
    rotate_angle = osd_info.get('rotate', 0)
    print(rotate_angle)
    
    if rotate_angle != 0:
        print(f"Rotating image by {rotate_angle} degrees")
        rotate_and_save_image(img_path, rotate_angle, out_path)
    else:
        print("Image is already correctly oriented")
    
    osd_info_after_rotation = pytesseract.image_to_osd(Image.open(out_path), output_type=Output.DICT)
    print(f"OSD Info after rotation: {osd_info_after_rotation}")
    
    rotate_angle_after = osd_info_after_rotation.get('rotate', 0)
    print(rotate_angle_after)
    
    if rotate_angle_after != 0:
        print(f"Rotating image again by {rotate_angle_after} degrees")
        rotate_and_save_image(out_path, rotate_angle_after, out_path)
    else:
        print("Image is correctly oriented after the second check")

def main():
    detect_and_rotate_image(args.img_path, args.out_path)
    
if __name__ == "__main__":
    main()

