from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim

class LowPassTransform(object):
    def __init__(self, threshold=45, probability=0.5):
        self.treshold = threshold
        self.probability = probability

    def low_pass(self, img, size):
        h, w = img.shape[0:2]#Getting image properties
        h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
        img2 = np.zeros((h, w), np.uint8)#Define a blank black image with the same size as the Fourier Transform Transfer
        img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
        #Center point plus or minus half of the filter size,
        # # forming the filter coordinates, then set pixels within filter to 1, preserving the low frequency part
        return img2*img #A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.

    def __call__(self, sample):
        if torch.rand(1) > self.probability:
            return sample
        
        np.seterr(divide = 'ignore') 
        
        gray = np.asarray(sample.convert('L')) 
        # gray = sample
        # Fourier transform
        img_dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(img_dft)  # Move frequency domain from upper left to middle

        # Low-pass filter
        dft_shift = self.low_pass(dft_shift, self.treshold)
        res = np.log(np.abs(dft_shift))

        # Inverse Fourier Transform
        idft_shift = np.fft.ifftshift(dft_shift)  # Move the frequency domain from the middle to the upper left corner
        ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
        ifimg = np.abs(ifimg)
    
        np.seterr(divide = 'warn') 

        # res = np.expand_dims(np.uint8(ifimg), axis=0)

        # return torch.tensor(res)
        return torch.tensor(np.uint8(ifimg))

class LowPassTransform3D(object):
    def __init__(self, threshold=45, probability=0.5):
        self.treshold = threshold
        self.probability = probability

    def low_pass(self, img, size):
        h, w = img.shape[0:2]#Getting image properties
        h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
        img2 = np.zeros((h, w), np.uint8)#Define a blank black image with the same size as the Fourier Transform Transfer
        img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
        return img2*img #A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.

    def __call__(self, sample):
        if torch.rand(1) > self.probability:
            return sample
    
        np.seterr(divide = 'ignore') 

        array = np.asarray(sample.copy())

        for i in range(3):
            gray = array[i,:,:]
        
            gray = np.array(gray) 

            # Fourier transform
            img_dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(img_dft)  # Move frequency domain from upper left to middle

            # Low-pass filter
            dft_shift = self.low_pass(dft_shift, self.treshold)
            res = np.log(np.abs(dft_shift))

            # Inverse Fourier Transform
            idft_shift = np.fft.ifftshift(dft_shift)  # Move the frequency domain from the middle to the upper left corner
            ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
            ifimg = np.abs(ifimg)
        
            
            array[i,:,:] = np.uint8(ifimg) 
            
        np.seterr(divide = 'warn') 

        return torch.tensor(array)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # print (f"new lr: {lr}")


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
