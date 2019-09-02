# ***dogs_vs_cats***
Udacity Machine Learning Capstone Project.


# Libs imported in this project
os; numpy; pandas; pathlib; pprint; seaborn; functools; matplotlib; keras; sklearn; tensorflow; IPython; gc.


# Machine OS & Hardware
Machine: Amazon EC2(p2.xlarge instance); OS: Deep Learning AMI (Ubuntu) Version 23.0, Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-1088-aws x86_64v); Hardware: 4 vCPU, 1000G user filesystem, NVIDIA Tesla K80 GPUs, 12 GiB of memory per GPU.


# Code illustration
data_processing.ipynb: data preprocessing, including data visualization, data cleaning, spliting trainig&validation set, data augmentation; model_training_x.ipynb: including model building, model training with different optimizer & parameter, model prediction.


# Training Time used
model which combining Xception&InceptionV3: about 700s per epochs; model which combining NARSNetLarge&InceptionResNetV2: about 2580s per epochs.


# Data sources
refer to https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data.


# Additional
file 'cheat.csv' which contains some error-labeled images in training set, comes from kaggle forum.
