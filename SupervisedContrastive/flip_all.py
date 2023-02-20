from torchvision import transforms
from PIL import Image
import os

if __name__ == '__main__':
    transforms_list = [
    transforms.RandomHorizontalFlip(p=1),
    ]

    img_dir = "/dataset/all/C23_C24_pos-concat-horflip-origs/neg"
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        img = Image.open(img_file)
        
        for i, tr in enumerate(transforms_list):
            processed_img = tr(img)
            processed_img.save(f"{img_file}")

    img_dir = "/dataset/all/C23_C24_pos-concat-horflip-origs/poz"
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        img = Image.open(img_file)
        
        for i, tr in enumerate(transforms_list):
            processed_img = tr(img)
            processed_img.save(f"{img_file}")