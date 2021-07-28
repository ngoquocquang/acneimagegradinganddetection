import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

annotations = glob('E:/Document/Do An III/Joint Acne Image Grading and Counting/dataset/Annotations/*.xml')

df = []
cnt = 0

for file in annotations:
  prev_filename = file.split('/')[-1].split('\\')[-1].split('.')[0] + '.jpg'
  filename = str(cnt) + '.jpg'
  row = []
  parsedXML = ET.parse(file)

  for node in parsedXML.getroot().iter('object'):
      acne = node.find('name').text
      xmin = int(node.find('bndbox/xmin').text)
      xmax = int(node.find('bndbox/xmax').text)
      ymin = int(node.find('bndbox/ymin').text)
      ymax = int(node.find('bndbox/ymax').text)
      row = [prev_filename, acne, xmin, xmax, ymin, ymax]
      df.append(row)
  cnt += 1

data = pd.DataFrame(df, columns=['filename','name', 'xmin', 'xmax', 'ymin', 'ymax'])

data[['filename', 'name', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('acne_detection.csv', index=False)
print('Done!')