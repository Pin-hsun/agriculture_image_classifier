import os
from os.path import join
import datetime
import argparse
import json
import torch
from torch import nn
from torch import optim
from torch.utils import data
from tensorboardX import SummaryWriter
from helpers import Progressbar, add_scalar_dict
from model.model import efficientnet, densenet_201, Resnext
from usage_metrics.Metric import Metric
from dataloader import AgriDataset

network_map = {
    'meta_densenet_201': densenet_201,
    'meta_efficientnet': efficientnet,
    'meta_resnext': Resnext
}

class Classifier():
    """
    Classifier for agriculture images
    class = [
          'asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
          'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce',
          'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
          'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo'
     ]
    """
    def __init__(self, gpu, lr, net):
        self.gpu = gpu
        self.model = network_map[net]()
        self.model.train()
        if gpu:
            self.model.cuda()
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def train_model(self, img_a, label, geo_info, metric): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        for p in self.model.parameters():
            p.requires_grad = True
        # enable_running_stats(self.model)
        pred = self.model(img_a, geo_info)
        _, label = label.max(1)
        label = label.type(torch.int64)

        dc_loss = self.loss(pred, label)
        self.optimizer.zero_grad()
        dc_loss.backward()
        self.optimizer.step()

        _, predicted = pred.max(1)
        metric.update(predicted, label)
        acc = metric.accuracy()
        f1 = metric.f1()

        errD = {
            'd_loss': dc_loss.mean().item()
        }
        return errD, acc, f1

    def eval_model(self, img_a, label, geo_info, metric): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        with torch.no_grad():
            pred = self.model(img_a, geo_info)
        label = label.type(torch.float)

        _, predicted = pred.max(1)
        _, targets = label.max(1)
        metric.update(predicted, targets)
        acc = metric.accuracy()
        f1 = metric.f1()
        each_f1 = metric.f1(each_cls=True)

        return acc, f1, each_f1

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        states = {
            'model': self.model.state_dict(),
        }
        torch.save(states, path)


if __name__=='__main__':
    dst_root = '/media/ExtHDD02/argulture_log/' # model saving
    data_root = '/media/ExtHDD01/Dataset/argulture/' # image data
    csv_path = '/media/ExtHDD01/Dataset/argulture/final.csv' # ID & labels

    # set model parameters
    args = {'gpu':True,
            'prj':'prj_name',
            'epochs':30,
            'batch_size_per_gpu':3,
            'lr':0.02,
            'net':'meta_densenet_201',
    }

    os.makedirs(join(dst_root, args['prj']), exist_ok=True)
    os.makedirs(join(dst_root, args['prj'], 'checkpoint'), exist_ok=True)
    writer = SummaryWriter(join(dst_root, args['prj'], 'summary'))

    with open(join(dst_root, args['prj'], 'setting.txt'), 'w') as f:
        f.write(json.dumps(args, indent=4, separators=(',', ':')))

    train_dataset = MetaDataset(data_root, csv_path, 'train', 0)
    valid_dataset = MetaDataset(data_root, csv_path, 'valid', 0)
    test_dataset = MetaDataset(data_root, csv_path, 'test', 0)

    num_gpu = torch.cuda.device_count()

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args['batch_size_per_gpu'] * num_gpu,
                                       num_workers=10, drop_last=True)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=args['batch_size_per_gpu'] * num_gpu, shuffle=False, num_workers=10, drop_last=False)

    print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

    classifier = Classifier(gpu=args['gpu'], lr=args['lr'], net=args['net'])
    progressbar = Progressbar()

    it = 0
    it_per_epoch = len(train_dataset) // (args['batch_size_per_gpu'] * num_gpu)
    for epoch in range(args['epochs']):
        lr = args['lr'] * (0.9 ** epoch)
        classifier.set_lr(lr)
        classifier.train()
        writer.add_scalar('LR/learning_rate', lr, it + 1)
        metric_tr = Metric(num_classes=33)
        metric_ev = Metric(num_classes=33)
        for img_a, att_a, geo_info, _ in progressbar(train_dataloader):
            img_a = img_a.cuda() if args['gpu'] else img_a
            att_a = att_a.cuda() if args['gpu'] else att_a
            geo_a = geo_info.cuda() if args['gpu'] else geo_info
            att_a = att_a.type(torch.float)
            img_a = img_a.type(torch.float)
            geo_a = geo_a.type(torch.float)

            errD, acc, f1 = classifier.train_model(img_a, att_a, geo_a, metric_tr)
            add_scalar_dict(writer, errD, it+1, 'D')
            it += 1
            progressbar.say(epoch=epoch, d_loss=errD['d_loss'], acc=acc.item(), f1=f1.item())
        classifier.save(os.path.join(
            dst_root, args['prj'], 'checkpoint', 'weights.{:d}.pth'.format(args['epoch'])
        ))


# training code
# CUDA_VISIBLE_DEVICES=3 python main_first.py --net meta_densenet --experiment_name meta_densenet_sam_optim_512_1028_1030 --lr 0.0005 --gpu
# fine tune code
# CUDA_VISIBLE_DEVICES=3 python main_first.py --net meta_densenet --experiment_name meta_densenet_sam_optim_512_1028_1030 --lr 0.0005 --ckpt meta_densenet_sam_optim_512_1028/checkpoint/weights.29.pth --gpu --batch_size 15