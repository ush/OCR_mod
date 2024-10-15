import os
import random
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--out_path', type=str, default=None)
args = parser.parse_args()

def get_centered_region(img_shape, region_size_range):
    h, w = img_shape[:2]
    center_x, center_y = w // 2, h // 2
    max_offset_x = w // 4
    max_offset_y = h // 4

    x = random.randint(int(center_x - 1.4*max_offset_x), int(center_x + 0.6*max_offset_x - region_size_range[1]))
    y = random.randint(center_y - max_offset_y, center_y + max_offset_y - region_size_range[1])
    
    w_region = random.randint(region_size_range[0], region_size_range[1])
    h_region = random.randint(region_size_range[0], region_size_range[1])

    return x, y, w_region, h_region

def apply_blur(img, regions):
    for region in regions:
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)
        img[y:y+h, x:x+w] = blurred_roi
    return img

def apply_random_shapes(img, num_shapes=8):
    h, w, _ = img.shape
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'circle'])
        color = [0, 0, 0]
        thickness = -1

        x, y, shape_w, shape_h = get_centered_region(img.shape, (100, 200))

        if shape_type == 'rectangle':
            cv2.rectangle(img, (x, y), (x + shape_w, y + shape_h), color, thickness)
        elif shape_type == 'circle':
            radius = random.randint(50, 100)
            center = (x + radius, y + radius)
            cv2.circle(img, center, radius, color, thickness)

    return img

def apply_dark_regions(img, regions):
    for region in regions:
        x, y, w, h = region
        img[y:y+h, x:x+w] = img[y:y+h, x:x+w] * 0.5  # Darken region
    return img

def corrupt_image(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)

    blur_regions = [get_centered_region(img.shape, (100, 200)) for _ in range(6)]
    img = apply_blur(img, blur_regions)

    img = apply_random_shapes(img)

    cv2.imwrite(output_image_path, img)

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            corrupt_image(input_image_path, output_image_path)
            print(f'Processed {filename}')


def main():
    process_images(args.img_path, args.out_path)

if __name__ == "__main__":
    main()

