import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os

img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
img_keys = 'TrachomaData/trachomagroundtruthkeyWungrades.csv'

keys_df = pd.read_csv(img_keys, sep=',', header=0, usecols=['imagename', 'ans_ICAPS'])

red = np.zeros([256, 1])
green = np.zeros([256, 1])
blue = np.zeros([256, 1])

hist_grade = [blue, green, red]

red_ungrade = np.zeros([256, 1])
green_ungrade = np.zeros([256, 1])
blue_ungrade = np.zeros([256, 1])

hist_ungrade = [blue_ungrade, green_ungrade, red_ungrade]

# loop over the input images
for r in keys_df.iterrows():
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    imagePath = os.path.join(img_dir, '0' + r[1]['imagename']) + '.jpg'
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for c in range(1):
        # hist = cv2.calcHist(image[:, :, c], [0], mask=None, histSize=[256], ranges=[0, 256])
        hist = cv2.calcHist(gray, [0], mask=None, histSize=[256], ranges=[0, 256])

        if r[1]['ans_ICAPS'] == 'Ungradeable':
            hist_ungrade[c] += hist
        else:
            hist_grade[c] += hist


fig, ax = plt.subplots(3, 1, sharex=True)

for c in range(1):
    ax[c].plot(hist_grade[c]/sum(hist_grade[c]))
    ax[c].plot(hist_ungrade[c] / sum(hist_ungrade[c]))

