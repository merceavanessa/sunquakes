import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np

from torchvision import transforms,models
from torchsummary import summary

from util import LowPassTransform3D, TwoCropTransform

from networks.resnet_big import SupConResNet
from dataset_val import MyDatasetVal

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')

    parser.add_argument('--emb_test', type=str, default='test_emb1')
    
    parser.add_argument('--model', type=str, default='resnet18')

    parser.add_argument('--data_folder', type=str, default='', help='path to custom dataset')
    parser.add_argument('--ckpt', type=str, default='',
                            help='path to pre-trained model')
    
    parser.add_argument('--syncBN', action='store_true',
                    help='using synchronized batch normalization')


    parser.add_argument('--encoding_size', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=128)

    opt = parser.parse_args()
    return opt


def set_loader(opt):
    rand_pos = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=[90,90]),
            transforms.RandomRotation(degrees=[180, 180]),
            transforms.RandomRotation(degrees=[270, 270]),
        ])

    erase_transform = transforms.RandomApply([
                    transforms.RandomErasing(p=0.05, scale=(0.04, 0.05), ratio=(0.2,0.6), value=100),
                    transforms.RandomErasing(p=0.05, scale=(0.04, 0.05), ratio=(0.3,0.6), value=100),
                    transforms.RandomErasing(p=0.1, scale=(0.03, 0.04), ratio=(0.4,0.6), value=100),
                    transforms.RandomErasing(p=0.1, scale=(0.03, 0.04), ratio=(0.5,0.5), value=100),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), ratio=(0.5,0.5), value=100),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), ratio=(0.4,0.6), value=100),
                    transforms.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.3,0.6), value=100),
                    transforms.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.2,0.6), value=100),
        ],p=1)
    
    lp = transforms.Compose([LowPassTransform3D(threshold=50, probability=1),
                                    transforms.RandomSolarize(threshold=50, p=1)])

    tens = transforms.Lambda(lambda x: torch.tensor(x))

                                    
    train_transform =  transforms.Compose([
                        transforms.Lambda(lambda x: np.transpose(np.asarray(x), (2,0,1))),
                        transforms.RandomChoice([lp, tens],p=[0.5,0.5]),
                        rand_pos,
                        # erase_transform
    ])
        
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: np.transpose(np.asarray(x), (2,0,1))),
        # transforms.RandomChoice([lp, tens],p=[1,0]),
        tens
    ])

    print("Loading custom dataset...")

    val_transform = TwoCropTransform(val_transform)

    val_dataset = MyDatasetVal(data_dir=opt.data_folder,
                                transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return val_loader

def initialize_model(opt, device, use_pretrained=True):
    if (opt.model =='resnet50'):
        model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft.fc = nn.Linear(2048, opt.encoding_size)
    else:
        model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft.fc = nn.Linear(512, opt.encoding_size)
            
    weight = model_ft.conv1.weight.clone()
    model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        model_ft.conv1.weight[:, :3] = weight

    model = SupConResNet(model_ft, name=opt.model,encoder_dim=opt.encoding_size, feat_dim=opt.latent_dim)
    
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    # if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
    model.encoder = torch.nn.DataParallel(model.encoder)
        # cudnn.benchmark = True
    
    # print (summary(model,(3,224,224)))
    if opt.ckpt != '':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        optimizer_state_dict = ckpt['optimizer']
        model.load_state_dict(state_dict)
    
   
    model = model.to(device)

    return model

def main():
    opt = parse_option()
    if opt.encoding_size == 0:
        if opt.model == 'resnet18':
            opt.encoding_size = 512
        else: 
            opt.encoding_size = 2048 
    # else: # comentat pentru a rula exemplul din 07.07
        # opt.latent_dim = opt.encoding_size


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    val_loader = set_loader(opt)
    
    model  = initialize_model(opt, device, use_pretrained=True)
    model.eval()

    print("Performing inference...")
    
    target_file = f"./out/{opt.emb_test}.csv"
            
    df = pd.DataFrame()
    for idx, item in enumerate(val_loader):
        (images, labels, names) = item
            
        if torch.cuda.is_available():
            image = images[0].float().to(device)
        else:
            image = images[0].float()
        
        bsz = labels.shape[0]
        with torch.set_grad_enabled(False):
            features = model.encoder(image)

        new_df = pd.DataFrame(data=features.view(bsz, -1).cpu().detach().numpy())
        new_df['label'] = labels.cpu().detach().numpy()
        new_df['name'] = names
        
        df = pd.concat([df,new_df])

    df.to_csv(target_file, index=False)


if __name__ == '__main__':
    main()
