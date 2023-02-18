import pandas as pd
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms

def read_data_set(data_dir):
    all_img_files = []
    all_labels = []

    class_names = os.walk(data_dir).__next__()[1]

    for index, class_name in enumerate(class_names):
        label = 0 if class_name == "neg" else 1
        img_dir = os.path.join(data_dir, class_name)
        img_files = os.walk(img_dir).__next__()[2]
        img_files = sorted(img_files)

        for img_file in img_files:
            img_file = os.path.join(img_dir, img_file)
            all_img_files.append(img_file)
            all_labels.append(label)

    return np.array(all_img_files), np.array(all_labels)

def get_concats(data_dir):
    imgss, labelsS = read_data_set(data_dir)

    df = pd.DataFrame(data={'name': imgss, 'label' : labelsS})

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
        
    grouping = df.groupby(by=['event_id'])

    groups = [grouping.get_group(x) for x in grouping.groups]
    
    print(f"Total events: {len(groups)}")
    random.Random(1265).shuffle(groups)

    df = pd.concat(groups)

    print(df['date'].unique())
    print(f"positives: {len(df[df['label']==1])}")
    print(f"negatives: {len(df[df['label']==0])}")

    data = []
    for i, group in enumerate(groups):
        for i in range(1,len(group)-1):
            row = group.iloc[i]
            prev_event = group.iloc[i-1]
            next_event = group.iloc[i+1]

            if (dataset_has_upsampling and ((prev_event['transform']!= row['transform']) or (row['transform'] != next_event['transform']))):
                continue

            (img, label) = row['name'], row['label']

            img_and_prevs = [prev_event['name'], img, next_event['name']]
            data.append((img_and_prevs, label)) 

    random.Random(1265).shuffle(data)
    # imsh, labsh = zip(*data)
    # imgs, labels = list(imsh), list(labsh)
    print(f"Total train data count after prev composure: {len(data)}")

    return data

def gen_concats():
    source_data_dir = "/home/vanessa/Dev/DATASETS/C24_TEST_pos"
    target_data_dir = "/home/vanessa/Dev/DATASETS/C24_TEST_pos-concat"

    concats = get_concats(source_data_dir)

    for i, concat in enumerate(concats):
        label = concat[1]
        imgs = concat[0]

        curr_img_name = imgs[1].split('/')[-1]
        s = 224
        
        img_prev1 = Image.open(imgs[0])
        img_curr  = Image.open(imgs[1])
        img_next1 = Image.open(imgs[2])

        img_prev1 = np.asarray(img_prev1.convert('L').resize(size=(s,s)))
        img_curr = np.asarray(img_curr.convert('L').resize(size=(s,s)))
        img_next1 = np.asarray(img_next1.convert('L').resize(size=(s,s)))

        if (img_prev1.shape != img_next1.shape):
            print(img_prev1.shape,img_next1.shape)
        merged = np.stack([img_prev1, img_curr, img_next1], axis=0)
        merged = np.transpose(np.asarray(merged), (1,2,0))

        merged = transforms.ToPILImage()(merged)

        merged.save(f"{target_data_dir}/{'poz' if label == 1 else 'neg'}/{curr_img_name}")
        

if __name__ == '__main__':
    gen_concats()
