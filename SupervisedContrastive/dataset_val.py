import os
from typing import Optional, Sequence, Union
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
from PIL import Image

class MyDatasetVal(Dataset):
    def read_data_set(self):
        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_dir).__next__()[1]

        for _, class_name in enumerate(class_names):
            label = 0 if class_name == "neg" else 1
            img_dir = os.path.join(self.data_dir, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            img_files = sorted(img_files)

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                all_img_files.append(img_file)
                all_labels.append(label)

        return np.array(all_img_files), np.array(all_labels)

    def update_df_fields(self, df):
        dataset_has_upsampling = len(df[df['name'].str.contains('_T0')]) > 0
        
        if (dataset_has_upsampling):
            df[['useless1','useless2', 'useless3', 'part_of_quake','Year','month','day','Hour','Minute','Seconds','frame','transform']] = df['name'].str.split('_', expand=True)
          
            df.loc[~df['transform'].isna(), 'transform'] = df['transform'].str.extract('(\d+)', expand=False)
            df.loc[df['transform'].isna(),'transform'] = -1
            df['transform']=df['transform'].astype(int)
        else:
            df[['useless1','useless2', 'useless3', 'part_of_quake','Year','month','day','Hour','Minute','Seconds','frame']] = df['name'].str.split('_', expand=True)
            
        df['frame'] = df['frame'].str.extract('(\d+)', expand=False)
        df['frame']=df['frame'].astype(int)

        df['date'] = pd.to_datetime(df[['Year', 'month', 'day', 'Hour','Minute','Seconds']])
        df['event_id'] = df.groupby([df.date, df.part_of_quake]).ngroup()
        
        if (dataset_has_upsampling):
          df = df.sort_values(by=['date','transform','frame'])
        else:
          df = df.sort_values(by=['date','frame'])
            
        return df
            
    def __init__(self,
                 data_dir,
                 transform
                 ):
        self.data_dir = data_dir
        self.transforms = transform

        imgss, labelsS = self.read_data_set()
        self.labels = []

        df = pd.DataFrame(data={'name': imgss, 'label' : labelsS})

        df = self.update_df_fields(df)
        
        grouping = df.groupby(by=['event_id'])
        groups = [grouping.get_group(x) for x in grouping.groups]

        print(f"Total events: {len(groups)}")
        random.Random(1265).shuffle(groups)

        df_x_test = pd.concat(groups)

        print(f"Positives: {len(df_x_test[df_x_test['label']==1])}")
        print(f"Negatives: {len(df_x_test[df_x_test['label']==0])}")
        
        test_data = [(img, label) for (img, label) in list(zip(imgss,labelsS)) if img in df_x_test['name'].tolist()]
        random.Random(1265).shuffle(test_data)
        imsh, labsh = zip(*test_data)
        self.imgs, self.labels = list(imsh), list(labsh)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = default_loader(self.imgs[idx])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.labels[idx], self.imgs[idx]