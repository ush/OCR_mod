import cv2
import os
import subprocess
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--out_path', type=str, default=None)
args = parser.parse_args()

def complex_preprocess_image(image_path, out_path, scale_factor=2):
	image = cv2.imread(image_path)

	upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
	
	cv2.imwrite(out_path, upscaled_image)
	
	final_image_path = apply_grayscale_script(out_path)

	return final_image_path
	
def apply_grayscale_script(image_path):
	subprocess.run(["bash", "textcleaner", "-g", "-e", "normalize", "-f", "25", "-o", "20", "-s", "1", image_path, image_path])
	return image_path
	
def no_preprocess_image(image_path):
	image = cv2.imread(image_path)
	return image_path
	
def get_new_path(file_path, dir_path):
	file_name = get_file_name(file_path)
	os.makedirs(dir_path, exist_ok = True)
	new_image_path = dir_path + file_name
	return new_image_path
	
def get_file_name(file_path):
	file_name = Path(file_path).name
	return file_name

def main():
    complex_preprocess_image(args.img_path, args.out_path)
    
if __name__ == "__main__":
    main()

