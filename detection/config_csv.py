import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

train = pd.read_csv('acne_detection.csv')
print(train.head())

data = pd.DataFrame()
data['format'] = train['filename']

for i in range(data.shape[0]):
  data['format'][i] = 'E:/Document/Do An III/Joint Acne Image Grading and Counting/dataset/JPEGImages/' + data['format'][i]

for i in range(data.shape[0]):
  data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['name'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')