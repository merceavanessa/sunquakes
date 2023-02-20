import os
import argparse
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
import random
import tensorboard_logger as tb_logger
# from autoaugment import CIFAR10Policy

from torchvision import transforms,models, datasets
from torchsummary import summary

from util import LowPassTransform, LowPassTransform3D, TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate,save_model

from networks.resnet_big import SupConResNet
from losses import SupConLoss, WeightedSupConLoss
from dataset_3_ch import MyDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')  
    parser.add_argument('--optimizer', type=str, default="sgd",
                        help='optimizer')
    parser.add_argument('--lr_decay_epochs', type=str, default='500,600,700',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--ignore_optimizer_ckpt', type=bool, default=False,
                        help='ignore_optimizer_ckpt')

    parser.add_argument('--emb_test', type=str, default='embeddings_contrastive_test')
    parser.add_argument('--emb_train', type=str, default='embeddings_contrastive_train')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    parser.add_argument('--encoding_size', type=int, default=0, help='size of encoder embedding')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--feature_extract', type=bool, default=False)
    parser.add_argument('--no_upsampling', action='store_true')
    parser.add_argument('--sample_with_replacement', action='store_true')
    parser.add_argument('--autoaug', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='path',
                        choices=[ 'path','cifar10'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='/home/vanessa-m/Dev/Datasets/C23_C24_pos', help='path to custom dataset')
    parser.add_argument('--mode', type=str, default='finetuning', help='parameter mode (finetuning/classification)')
    parser.add_argument('--ckpt', type=str, default='',
                            help='path to pre-trained model')
    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
    parser.add_argument('--latent_size', type=int, default=128, help='latent size of embedding')
    parser.add_argument('--custom_tr', type=bool, default=True, help='apply custom astro specific transforms vs standard imagenet ones')

    # weighting
    parser.add_argument('--weighted', action='store_true', help='use weighted supcon loss')
     
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None 
        # \
        #     and opt.mean is not None \
        #     and opt.std is not None

    if opt.encoding_size == 0:
        if opt.model == 'resnet50':
            opt.encoding_size = 2048
        else:
            opt.encoding_size = 512

    if not opt.no_upsampling:
        print("Training on upsampled dataset")

    # set the path according to the environment
    print (opt.data_folder)
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = '/out/save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = '/out/save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_loader(opt):
    print (opt.dataset)
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = (1.2883e-06, 1.2883e-06, 1.2883e-06)
        std = (0.0005, 0.0005, 0.0005)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    rand_pos = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=[90,90]),
            transforms.RandomRotation(degrees=[180, 180]),
            transforms.RandomRotation(degrees=[270, 270])], p=0.5)
        ],p=[0.2,0.2,0.6])

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
        
    if opt.dataset == 'path':
        if opt.custom_tr: 
            lp = transforms.Compose([LowPassTransform3D(threshold=50, probability=1),
                                    transforms.RandomSolarize(threshold=50, p=1)])

            tens = transforms.Lambda(lambda x: torch.tensor(x))

                                    
            train_transform =  transforms.Compose([
                                transforms.Lambda(lambda x: np.transpose(np.asarray(x), (2,0,1))),
                                transforms.RandomChoice([lp, tens],p=[0.5,0.5]),
                                rand_pos,
                                erase_transform
            ])
                
            val_transform = transforms.Compose([
                transforms.Lambda(lambda x: np.transpose(np.asarray(x), (2,0,1))),
                tens
            ])
          
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                normalize
            ])
            
            val_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                normalize
            ])
    else:
        if (opt.autoaug):
            train_transform = transforms.Compose([
                # CIFAR10Policy(),
                transforms.ToTensor(),
                normalize
            ])
            
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            
        else:
            # our transforms on cifar
            lp = transforms.Compose([LowPassTransform3D(threshold=50, probability=1),
                                    transforms.RandomSolarize(threshold=50, p=1)])

            tens = transforms.Lambda(lambda x: torch.tensor(x))

                                    
            train_transform =  transforms.Compose([
                                transforms.Lambda(lambda x: np.transpose(np.asarray(x), (2,0,1))),
                                transforms.RandomChoice([lp, tens],p=[0.5,0.5]),
                                rand_pos,
                                erase_transform
            ])
                
            val_transform = transforms.Compose([
                transforms.Lambda(lambda x: np.transpose(np.asarray(x), (2,0,1))),
                tens
            ])
    

    if opt.dataset == 'path':
        print("Loading custom dataset...")

        if opt.mode == 'finetuning':
            train_transform = TwoCropTransform(train_transform)
            val_transform = TwoCropTransform(val_transform)

        train_dataset = MyDataset(data_dir=opt.data_folder, 
                                    split="train", 
                                    transform=train_transform)

        val_dataset = MyDataset(data_dir=opt.data_folder,
                                    split="test", 
                                    transform=val_transform)
    elif opt.dataset == 'cifar10':
        if opt.mode == 'finetuning':
            train_transform = TwoCropTransform(train_transform)
            val_transform = TwoCropTransform(val_transform)
            
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                train=False,
                                transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    if opt.sample_with_replacement:
        targets = train_dataset.labels
        class_sample_count = np.unique(targets, return_counts=True)[1]
        weight =  1. / class_sample_count 
        samples_weight = weight[targets]

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size,
            sampler=torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight),
             replacement=True),
            num_workers=opt.num_workers, 
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size,  shuffle=True,
            num_workers=opt.num_workers, 
            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.encoder.fc.parameters():
            param.requires_grad = True

        for param in model.head_class.parameters():
            param.requires_grad = True

        for param in model.encoder.layer4.parameters():
            param.requires_grad = True


def initialize_model(latent_size, feature_extract, opt, device, use_pretrained=True):
    if (opt.mode == 'finetuning'):
        if (opt.model =='resnet50'):
            model_ft = models.resnet50(pretrained=use_pretrained)
            model_ft.fc = nn.Linear(2048, opt.encoding_size)
        else:
            model_ft = models.resnet18(pretrained=use_pretrained)
            model_ft.fc = nn.Linear(512, opt.encoding_size)
            
        set_parameter_requires_grad(model_ft, feature_extract)

        weight = model_ft.conv1.weight.clone()
        model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            model_ft.conv1.weight[:, :3] = weight

        model = SupConResNet(model_ft, n_class=2 if opt.dataset=='path' else 10, feat_dim=latent_size, name=opt.model, encoder_dim=opt.encoding_size)
    else:
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

        model = SupConResNet(model_ft, n_class=2 if opt.dataset=='path' else 10, feat_dim=latent_size, name=opt.model, encoder_dim=opt.encoding_size)
        set_parameter_requires_grad(model, feature_extract)
    
    model, optimizer = set_optimizer_custom(opt, feature_extract, model, device)
    start_epoch = 0
    
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        cudnn.benchmark = True
    
    if opt.ckpt != '':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        optimizer_state_dict = ckpt['optimizer']
        model.load_state_dict(state_dict)
        model = model.to(device)
        if (not opt.ignore_optimizer_ckpt):
            optimizer.load_state_dict(optimizer_state_dict)
            start_epoch = ckpt['epoch']
            print(ckpt['epoch'])
                
    if (not opt.weighted):
        criterion = SupConLoss(temperature=opt.temp)
    else:               
        print("Using weighted loss")          
        criterion = WeightedSupConLoss(temperature=opt.temp)
    
    if (opt.dataset == 'path'):
        if (opt.weighted):
            weights = torch.FloatTensor([0.4, 1]).to(device)
            criterion_class = nn.CrossEntropyLoss(weight=weights,reduction='mean')
        else:
            criterion_class = nn.CrossEntropyLoss(reduction='mean')

    else:
        criterion_class = nn.CrossEntropyLoss(reduction='mean')
    
    model = model.to(device)
    criterion = criterion.to(device) 
    criterion_class = criterion_class.to(device) 
    
    # print(summary(model, (3,224,224)))

    return model, optimizer, criterion, criterion_class, start_epoch

def train_model(logger, model, dataloaders, criterion, criterion_class, optimizer, opt, device, start_epoch=0):
    losses = AverageMeter()
    
    val_acc_history = []
    num_epochs = opt.epochs

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    
    #early stopping
    last_loss = 100
    patience = 6
    trigger_times = 0

    since = time.time()
    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_class_loss = 0.0
            running_corrects = 0
            f1_scores = 0
            precision_scores = 0
            recall_scores = 0
            accuracy_scores = 0
            
            # Iterate over data.
            for idx, item in enumerate(dataloaders[phase]):
                if opt.dataset == 'path':
                    images, labels, _ = item
                else:
                    images, labels = item

                if opt.mode == 'finetuning':
                    inputs = torch.cat([images[0], images[1]], dim=0)
                    bsz=labels.shape[0]
                else:
                    inputs = images
                    bsz=labels.shape[0]
                    
                if (bsz != opt.batch_size):
                    continue
                
                if torch.cuda.is_available():
                    inputs = inputs.float().to(device)
                    labels = labels.to(device)
                    
                warmup_learning_rate(opt, epoch, idx, len(dataloaders[phase]), optimizer)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # with torch.autograd.detect_anomaly():
                    outputs, outputs_class = model(inputs)

                    if opt.mode == 'finetuning':
                        f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        if opt.method == 'SupCon':
                            if (opt.weighted):
                                if (not opt.no_upsampling):
                                    weights = [0.00015528 if x == 0 else 0.00030618 for _, x in enumerate(labels.tolist())]
                                else:
                                    weights = [0.0001377235031318902 if x == 0 else 0.0012340768981590972 for _, x in enumerate(labels.tolist())]
                                # weights = [0.71 if x == 0 else 6.17 for _, x in enumerate(labels.tolist())]
                                loss = criterion(features, labels, mask=None, weights=torch.tensor(weights))
                            else:
                                loss = criterion(features, labels)
                        elif opt.method == 'SimCLR':
                            loss = criterion(features)
                        else:
                            raise ValueError('contrastive method not supported: {}'.
                                            format(opt.method))
                        
                        outputs_class, _ =  torch.split(outputs_class, [bsz, bsz], dim=0)

                    average = 'binary' if opt.dataset == 'path' else 'macro'
                    
                    loss_class = criterion_class(outputs_class, labels)
                    preds = torch.argmax(outputs_class, 1)
                    cpu_labels = labels.data.cpu().detach().numpy()
                    cpu_preds = preds.data.cpu().detach().numpy()
                    f1_scores += f1_score(cpu_labels, cpu_preds, average=average, zero_division=0)
                    precision_scores += precision_score(cpu_labels, cpu_preds, average=average, zero_division=0)
                    recall_scores += recall_score(cpu_labels, cpu_preds, average=average, zero_division=0)
                    accuracy_scores += accuracy_score(cpu_labels, cpu_preds)

                    if opt.mode == 'finetuning':
                        losses.update(loss.item(), opt.batch_size)
                    else:
                        losses.update(loss_class.item(), opt.batch_size)

                    if phase == 'train':
                        optimizer.zero_grad()
                        if opt.mode == 'finetuning':
                            loss.backward()
                        else:
                            loss_class.backward()
                        optimizer.step()

                    if (idx + 1) % opt.print_freq == 0:
                     print('Train: [{0}][{1}/{2}]\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                        epoch, idx + 1, len(dataloaders[phase]), loss=losses))

                    # statistics
                    if opt.mode=='finetuning':
                        running_loss += loss.item() # #* inputs.size(0)
                    running_class_loss += loss_class.item()  #* inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_class_loss = running_class_loss / len(dataloaders[phase])

            logger.log_value(f'{phase}_loss', epoch_loss, epoch)
            logger.log_value(f'{phase}_class_loss', epoch_class_loss, epoch)

            epoch_acc = accuracy_scores / len(dataloaders[phase])
            epoch_f1 = f1_scores / len(dataloaders[phase])
            epoch_prec = precision_scores / len(dataloaders[phase])
            epoch_rec = recall_scores / len(dataloaders[phase])
            print('{} Contrastive Loss: {:.4f} Class Loss: {:.4f} Acc: {:4f}, F1: {:4f}, Prec: {:4f}, Rec: {:4f}'.format(phase, epoch_loss, epoch_class_loss, epoch_acc, epoch_f1, epoch_prec, epoch_rec))

            logger.log_value(f'{phase}_accuracy', epoch_acc, epoch)
            logger.log_value(f'{phase}_precision', epoch_prec, epoch)
            logger.log_value(f'{phase}_recall', epoch_rec, epoch)
            logger.log_value(f'{phase}_f1', epoch_f1, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            if phase == 'val':
                if epoch_loss > last_loss:
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)

                    if trigger_times >= patience:
                        print('Early stopping!\nStart to test process.')
                        return model
                else:
                    print('trigger times: 0')
                    trigger_times = 0

                last_loss = epoch_loss


            if epoch !=0 and epoch % opt.save_freq == 0:
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
            # deep copy the model
            # if opt.mode != 'finetuning':
            if phase == 'val' and epoch_acc > best_acc and epoch_f1 > best_f1:
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if opt.mode == 'finetuning':
        model.load_state_dict(best_model_wts)
    
    return model

def set_optimizer_custom(opt, feature_extract, model, device):
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
            else:
                print("\t not updating: ", name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                continue
                # print("\t",name)
            else:
                print("\t not updating: ", name)

    if opt.optimizer == 'sgd':
        optimizer_ft = optim.SGD(params_to_update,  lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)
    else: 
        optimizer_ft = optim.Adam(params_to_update,  lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)

    return model, optimizer_ft

def main():
    opt = parse_option()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    feature_extract = opt.feature_extract
    train_loader, val_loader = set_loader(opt)
    model, optimizer, criterion, criterion_class, start_epoch = initialize_model(opt.latent_size, feature_extract, opt, device, use_pretrained=True)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training/validation routine
    model = train_model(logger, model, {'train': train_loader, 'val': val_loader}, criterion, criterion_class, optimizer, opt, device, start_epoch)

    # save the last model
    if opt.epochs != 0:
        save_file = os.path.join(
        opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, opt.epochs, save_file)

    print("Performing inference...")
    loaders = [{'loader': train_loader, 'phase': 'train'},{'loader' : val_loader, 'phase': 'val'}]
    
    for _, loader in enumerate(loaders):
        if (loader['phase'] == 'train'):
            target_file = f"/out/{opt.emb_train}.csv"
        else:
            target_file = f"/out/{opt.emb_test}.csv"
            
        df = pd.DataFrame()
        for idx, item in enumerate(loader['loader']):
            if opt.dataset == 'path':
                (images, labels, names) = item
            else:
                images, labels = item 
                
            if opt.mode == 'finetuning':
                if torch.cuda.is_available():
                    image = images[0].float().to(device)
                
                bsz = labels.shape[0]
                with torch.set_grad_enabled(False):
                    features = model.encoder(image)
        
                new_df = pd.DataFrame(data=features.view(bsz, -1).cpu().detach().numpy())
            else:
                if torch.cuda.is_available():
                    image = images.float().to(device)
                    
                _, outputs_class = model(image)
                _, preds = torch.max(outputs_class, 1)
                new_df = pd.DataFrame(data={'predictions': preds.cpu().detach().numpy()})
                
            new_df['label'] = labels.cpu().detach().numpy()
            if opt.dataset == 'path':
                new_df['name'] = names
            
            df = pd.concat([df,new_df])

        df.to_csv(target_file, index=False)

if __name__ == '__main__':
  main()
