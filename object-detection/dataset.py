import os
import numpy as np
import torch
import torch.utils.data
import random
import pandas as pd
from PIL import Image
import datetime as dt

class SegDataset(torch.utils.data.Dataset):
    def read_data_set(self, mode='train', cycle='24'):
        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_dir).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = 0 if class_name == "neg" else 1
            if label == 0:
              continue

            img_dir = os.path.join(self.data_dir, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            img_files = sorted(img_files)

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                all_img_files.append(img_file)
                all_labels.append(label)

            
        df = pd.DataFrame(data={'name': all_img_files, 'label' : all_labels})#, 'event': events})

        df[['useless1', 'useless2', 'useless3', 'part_of_quake','Year','month','day','Hour','Minute','Seconds','frame']] = df['name'].str.split('_', expand=True)
    
        df['frame'] = df['frame'].str.extract('(\d+)', expand=False)
        df['frame']=df['frame'].astype(int)

        df['date'] = pd.to_datetime(df[['Year', 'month', 'day', 'Hour','Minute','Seconds']])
        df['event_id'] = df.groupby([df.date, df.part_of_quake]).ngroup()
        df = df.sort_values(by=['date','frame'])
            
        grouping = df.groupby(by=['event_id'])

        groups = [grouping.get_group(x) for x in grouping.groups]
        random.Random(1265).shuffle(groups)

        box_df = pd.read_csv(f"data_c{cycle}.csv")

        data = []
        test_thresh=0
        cnt=0
        for _, group in enumerate(groups):
          for i in range(0,len(group)):
            row = group.iloc[i]
            box_time = dt.datetime.strftime(row['date'],'%d.%m.%Y %H:%M',)

            boxes = []
            box_for_frame = box_df[box_df['Time'] == box_time]
            if (len(box_for_frame) > 0):
                box_for_frame = box_for_frame.iloc[0]
                if (box_for_frame['FS'] <= row['frame'] and box_for_frame['FE'] >= row['frame']):
                    for k in range(1,box_for_frame['box_cnt']+1):
                        box = box_for_frame[[f"b{k}_x0",f"b{k}_y0",f"b{k}_x1",f"b{k}_y1"]]
                        # boxes = np.append(np.array(box), boxes, axis=0) #only add boxes for positives
                        boxes.append(box)
  
            boxes = np.array(boxes)

            if (mode=='train' and cnt <= test_thresh) or (mode=='test' and cnt >= test_thresh):
              if (len(boxes) > 0):
               data.append((row['name'], row['label'], boxes))
          cnt+=1

        random.Random(1265).shuffle(data)
        imsh, labsh, boxesh = zip(*data)
        self.imgs, self.labels, self.boxes = list(imsh), list(labsh), list(boxesh)

    def __init__(self, root, mode='train', transforms=None, cycle='23'):
        self.data_dir = root
        self.transforms = transforms
        self.cycle = cycle

        self.read_data_set(mode, cycle)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # print(img_path)
        img = Image.open(img_path).convert("RGB")

        if (len(self.boxes[idx]) > 0):
          boxes = np.vstack(self.boxes[idx]).astype(float)
        else:
          boxes = torch.empty([0,4])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class (quake object)
        labels = torch.ones((2,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        if (len(self.boxes[idx])!= 0):
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0])
            
        # suppose all instances are not quake; can't rename this as it's used in the utils
        iscrowd = torch.zeros((2,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        # target["isquake"] = isquake
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)