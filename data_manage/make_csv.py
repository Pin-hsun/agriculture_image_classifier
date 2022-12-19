import csv
import pandas as pd
import os
import glob
import shutil
from zipfile import ZipFile


def mk_label_file(df, root):
    """save label csv file"""
    for current_type in os.listdir(root):
        if current_type.__contains__('csv'):
            pass
        else:
            for current_id in os.listdir(root+current_type):
                df.loc[df['Img'] == current_id, 'labels'] = current_type
                
    return df

def unzip(path_to_zip_file, directory_to_extract_to):
    with ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


if __name__=='__main__':
    train_root = ''
    eval_root = ''
    test_root = ''
    all_df = pd.read_csv(train_root + 'agriculture.csv', encoding='unicode_escape')
    all_df['labels'] = 'x'
    test_df = pd.read_csv(test_root + 'test.csv', encoding='unicode_escape')
    test_df['labels'] = 'x'
    eval_df = pd.read_csv(eval_root + 'eval.csv', encoding='unicode_escape')
    eval_df['labels'] = 'x'

    all_df = mk_label_file(all_df, train_root)
    all_df.to_csv(train_root + 'train.csv')

    all_df = pd.concat([eval_df, test_df])
    all_df.to_csv(test_csv + 'test.csv')

    os.makedirs(test_root + 'eval/', exist_ok=True)
    for i in glob.glob(eval_root+'*/*'):
        id = i.split('/')[-1]
        shutil.copy(i, test_root + 'eval/' + id)

