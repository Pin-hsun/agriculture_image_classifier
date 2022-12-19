import os
import cv2
import csv
import torch
import numpy as np
from torch import nn
from torch.utils import data
from dataloader import TestAgriDataset
from model.model import efficientnet, densenet_201, Resnext

# parameters
attrs_default = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
                 'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce',
                 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
                 'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo'
                 ]

# define functions
def network_map(net):
    network_mapping = {
        'meta_densenet_201': densenet_201(),
        'meta_efficientnet': efficientnet(),
        'meta_resnext': Resnext(),
        }

    return network_mapping[net]

def rot90(img):
    img = np.array(img.detach().cpu())
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    img_rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_rot90 = np.transpose(img_rot90, (2, 0, 1))

    return torch.Tensor(img_rot90).cuda()

def flip(img, dic='h'):
    img = np.array(img.detach().cpu())
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    if dic == 'h':
        fliped_img = cv2.flip(img, 1)
    elif dic == 'v':
        fliped_img = cv2.flip(img, 0)
    fliped_img = np.transpose(fliped_img, (2,0,1))

    return torch.Tensor(fliped_img).cuda()

def schedule(model, img, meta, model2, model3):
    final_out = torch.empty(1, 33).cuda()
    for i in range(img.shape[0]):
        original_img = img[i, ...]
        original_meta = meta[i, ...]
        original_meta = torch.unsqueeze(original_meta, 0)
        horizontal_flipped_img = flip(original_img, 'h')
        vertical_flipped_image = flip(original_img, 'v')
        horizontal_vertical_flipped_image = flip(vertical_flipped_image, 'h')
        original_img = torch.unsqueeze(original_img, 0)
        horizontal_flipped_img = torch.unsqueeze(horizontal_flipped_img, 0)
        vertical_flipped_image = torch.unsqueeze(vertical_flipped_image, 0)
        horizontal_vertical_flipped_image = torch.unsqueeze(horizontal_vertical_flipped_image, 0)

        rotate_img = rot90(img[i,::])
        rotate_horizontal_flipped_img = flip(rotate_img, 'h')
        rotate_vertical_rotate_img = flip(rotate_img, 'v')
        rotate_vertical_horizontal_flipped_img = flip(rotate_vertical_rotate_img, 'h')
        rotate_img = torch.unsqueeze(rotate_img, 0)
        rotate_horizontal_flipped_img = torch.unsqueeze(rotate_horizontal_flipped_img, 0)
        rotate_vertical_rotate_img = torch.unsqueeze(rotate_vertical_rotate_img, 0)
        rotate_vertical_horizontal_flipped_img = torch.unsqueeze(rotate_vertical_horizontal_flipped_img, 0)

        batch1_img = torch.cat([original_img, horizontal_flipped_img, vertical_flipped_image, horizontal_vertical_flipped_image, rotate_img, rotate_horizontal_flipped_img, rotate_vertical_rotate_img, rotate_vertical_horizontal_flipped_img], 0).cuda()
        original_meta = torch.cat([original_meta]*8, 0).cuda()
        softmax = nn.Softmax(dim=1)
        out = model(batch1_img, original_meta)
        out = softmax(out)
        out[:, 4] = out[:, 4] + 0.1
        out[:, 19] = out[:, 19] + 0.1
        if model2 is not None:
            try:
                out2 = model2(batch1_img)
            except:
                out2 = model2(batch1_img, original_meta)
            out2 = softmax(out2)
            out[:, 4] = out[:, 4] + 0.2
            out += out2

        if model3 is not None:
            try:
                out3 = model3(batch1_img)
            except:
                out3 = model3(batch1_img, original_meta)
            out3 = softmax(out3)
            out[:, 4] = out[:, 4] + 0.2
            out += out3
        out = torch.unsqueeze(torch.sum(out, 0), 0)
        final_out = torch.cat([final_out, out], 0)

    return final_out[1:, :]


if __name__=='__main__':
    root = ''
    data_root = ''
    csv_path = ''
    ckpt = 'checkpoint'

    # set testing parameters
    batch_size = 3
    net1 = 'meta_efficientnet'
    net2 = 'meta_densenet'
    net3 = 'meta_efficientnet'
    log_name1 = ''
    log_name2 = ''
    log_name3 = ''
    epoch1 = 1
    epoch2 = 1
    epoch3 = 1
    gpu = True

    test_dataset = TestAgriDataset(data_root, csv_path, 'test')
    num_gpu = torch.cuda.device_count()
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size * num_gpu, shuffle=False, num_workers=10, drop_last=False)
    print('Training images:', len(test_dataset))

    model = network_map(net1).cuda()
    states = torch.load(os.path.join(root, log_name1, ckpt, f'weights.{epoch1}.pth'))
    model.load_state_dict(states['model'])
    model.eval()

    if net2 is not None:
        model2 = network_map(net2).cuda()
        states = torch.load(os.path.join(root, log_name2, ckpt, f'weights.{epoch2}.pth'))
        model2.load_state_dict(states['model'])
        model2.eval()
    else:
        model2 = None
    if net3 is not None:
        model3 = network_map(net3).cuda()
        states = torch.load(os.path.join(root, log_name3, ckpt, f'weights.{epoch3}.pth'))
        model3.load_state_dict(states['model'])
        model3.eval()
    else:
        model3 = None

    id_t = ()
    pred_t = []
    for img, meta, id in test_dataloader:
        id_t += id
        img = img.cuda() if gpu else img
        meta = meta.cuda() if gpu else meta
        img = img.type(torch.float)
        meta = meta.type(torch.float)
        with torch.no_grad():
            pred = schedule(model, img, meta, model2, model3)
            _, predicted = pred.max(1)
            pred_t.append(predicted)

    pred = torch.cat(pred_t, 0)

    # save result
    os.makedirs('result', exist_ok=True)
    with open(f'result/{args.log_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i in range(len(pred)):
            tmp = [id_t[i], attrs_default[pred[i].item()]]
            writer.writerow(tmp)
