# agriculture_image_classifier
# Install
The list of libraries that you need to install to execute the code:
- python = 3.6
- pytorch = 0.4
- torchvision
- os
- pandas
- datetime
- json
- glob
- shutil
- zipfile
- numpy
- csv
- cv2
- tensorboardX

# Data preprocessing
- download the dataset and read the train/eval/test dataset csv in data_manage/make_csv.py
- resize images to 800x800
- random crop to 712x712
- normalization


# Training
We train three models(EfficientNet, ResNet and DenseNet) and ensemble.
use metrics (acc, f1 score) in usage_metrics/Metric to do validation
`python main.py`

# Testing
save result to csv file
`python test.py`