import os
import math
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset


def attrs_collect(dataset):
    if dataset == 0:
        attrs_all = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
                     'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce',
                     'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
                      'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo'
                      ]
    elif dataset == 1:
        attrs_all = ['broccoli', 'cauliflower', 'kale']

    return attrs_all


def split_train_test(root, type_list):
    front_total_len = 0
    train_set = list()
    valid_set = list()
    test_set = list()
    for one_root in type_list:
        one_path = root + one_root
        if os.path.isdir(one_path):
            total_len = len(os.listdir(one_path))
            train = math.ceil(total_len*1)
            valid = math.ceil(total_len*0)
            test = total_len - train - valid
            now_total_train = front_total_len + train
            now_valid_len = front_total_len + train + valid
            now_test_len = front_total_len + train + valid + test
            train_set += list(range(front_total_len, now_total_train))
            valid_set += list(range(front_total_len+train, now_valid_len))
            test_set += list(range(front_total_len+train+valid, now_test_len))
            front_total_len+=total_len

    return train_set, valid_set, test_set


class AgriDataset(Dataset):
    def __init__(self, data_root, csv_path, mode, dataset):
        attrs = attrs_collect(dataset)
        all_data_root = [x for x in attrs if os.path.isdir(data_root + x)] # farm type
        self.all_data_id = [data_root + one_data_root + '/' + x for one_data_root in all_data_root for x in os.listdir(data_root+one_data_root)]
        self.data_info = pd.read_csv(csv_path)
        self.label = pd.concat([self.data_info.iloc[:, 2:5], self.data_info.iloc[:, 7:9], self.data_info.iloc[:, 9:]], axis=1)
        self.mode = mode
        self.dataset = dataset
        train_set, valid_set, test_set = split_train_test(data_root, attrs)
        if mode == 'train':
            self.ids = train_set
        elif mode == 'valid':
            self.ids = valid_set
        elif mode == 'test':
            self.ids = test_set

    def transforms(self, mode, x_new, y_new):
        if mode == 'train':
            transform = A.Compose([
                A.Resize(width=y_new, height=x_new),
                A.RandomCrop(width=768, height=768),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(),
                A.Rotate(limit=15),
                A.CoarseDropout(max_holes=10)
                ])
        else:
            transform = A.Compose([
                A.Resize(width=y_new, height=x_new),
                A.CenterCrop(width=768, height=768),
                ])

        return transform

    def __getitem__(self, index):
        img = cv2.imread(self.all_data_id[self.ids[index]])
        id = self.all_data_id[self.ids[index]].split('/', 5)[-1].split('.')[0]

        x, y = img.shape[0], img.shape[1]
        if x > y:
            ratio = 780/y
            x_new = int(ratio * x)
            y_new = 780
        else:
            ratio = 780/x
            y_new = int(ratio * y)
            x_new = 780
        transform = self.transforms(self.mode, x_new, y_new)
        label = np.array(self.label[self.label['Img'] == self.all_data_id[self.ids[index]].rsplit('/')[-1]].iloc[:, 6:]).squeeze(axis=0)
        geo_info = np.array(self.label[self.label['Img'] == self.all_data_id[self.ids[index]].rsplit('/')[-1]].iloc[:, 3:5]).squeeze(axis=0)
        geo_info = np.concatenate([geo_info, np.ones(1)], 0)
        transformed = transform(image=img)
        img = np.transpose(transformed["image"], (2,0,1))
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5
        geo_value = [[121, 1.5], [23.5, 2], [1, 1]]
        for i in range(len(geo_info)):
            geo_info[i] = (geo_info[i] - geo_value[i][0])/geo_value[i][1]

        return img, label, geo_info, id

    def __len__(self):
        return len(self.ids)

    def get_labels(self):
        attrs = attrs_collect(dataset=self.dataset)
        search_dict = dict()
        for idx, attrs_type in enumerate(attrs):
            search_dict[attrs_type] = idx

        all_label = [search_dict[x] for x in self.label['labels']]
        _label = [all_label[x] for x in self.ids]

        return _label


class TestAgriDataset(Dataset):
    def __init__(self, data_root, csv_path, mode):
        all_data_root = [x for x in os.listdir(data_root) if os.path.isdir(data_root + x)] # farm type
        self.all_data_id = [data_root + one_data_root+'/'+x for one_data_root in all_data_root for x in os.listdir(data_root+one_data_root)]

        self.data_info = pd.read_csv(csv_path)
        self.label = pd.concat([self.data_info.iloc[:,2], self.data_info.iloc[:,7:]], axis=1)

        self.mode = mode

    def transforms(self, x_new, y_new):
        transform = A.Compose([
                A.Resize(width=y_new, height=x_new),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(),
                A.RandomCrop(width=768, height=768),
                ])

        return transform

    def __getitem__(self, index):
        img = cv2.imread(self.all_data_id[index])

        id = self.all_data_id[index].split('/')[-1]

        x, y = img.shape[0], img.shape[1]
        if x > y:
            ratio = 780/y
            x_new = int(ratio * x)
            y_new = 780
        else:
            ratio = 780/x
            y_new = int(ratio * y)
            x_new = 780
        transform = self.transforms(self.mode, x_new, y_new)
        transformed = transform(image=img)
        img = np.transpose(transformed["image"], (2, 0, 1))
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5
        geo_info = np.array(self.label[self.label['Img'] == id].iloc[:, 1:]).squeeze(axis=0)
        geo_info = np.concatenate([geo_info, np.ones(1)], 0)
        geo_value = [[121, 1.5], [23.5, 2], [1, 1]]
        for i in range(len(geo_info)):
            geo_info[i] = (geo_info[i] - geo_value[i][0])/geo_value[i][1]

        return img, geo_info, id

    def __len__(self):
        return len(self.all_data_id)