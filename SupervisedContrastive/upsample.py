from torchvision import transforms
from PIL import Image
import os

if __name__ == '__main__':
    transforms_list = [
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomRotation(degrees=[90,90]),
    transforms.RandomRotation(degrees=[180,180]),
    transforms.RandomRotation(degrees=[270,270])
    ]

    img_dir = '/dataset/C24_TEST_pos-upsampledall-concat/neg'#"/dataset/all/test_set_all-transformed-concat/neg"
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        img = Image.open(img_file)
        
        for i, tr in enumerate(transforms_list):
            processed_img = tr(img)
            processed_img.save(f"{img_file[:-4]}_T{i}.png")

    
    img_dir = '/dataset/C24_TEST_pos-upsampledall-concat/poz'#"/dataset/all/test_set_all-transformed-concat/neg"
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        img = Image.open(img_file)
        
        for i, tr in enumerate(transforms_list):
            processed_img = tr(img)
            processed_img.save(f"{img_file[:-4]}_T{i}.png")