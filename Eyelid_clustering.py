import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from sklearn.cluster import MiniBatchKMeans as cluster
from sklearn.decomposition import PCA
from torchvision import transforms

from DataLoader_lightning_mData import ToTensor, TrachomaDataModule

img_dir = 'm'
img_keys = 'm/tfti.csv'

trans_0 = transforms.Compose([ToTensor(),
    transforms.Resize(256),
     transforms.CenterCrop(224)])

dm = TrachomaDataModule(img_dir, img_keys, transforms_0=trans_0,
                        batch_size=1, num_workers=1, oversample=False, split=True)

dm.setup()
data_train = dm.train_dataloader()
data_test = dm.val_dataloader()

pca = PCA(n_components=2)

# images = glob.glob('m/*.jpg')
# im = cv2.imread(images[0])
# im = cv2.resize(im, (224, 224))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# pix = im.reshape(im.shape[0] * im.shape[1], 3)
# pix = pca.fit_transform(pix)
pix = None


for im in data_train:
    im = im['image'].squeeze()
    # im = cv2.imread(i)
    # im = cv2.resize(im, (224, 224))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2 = im.reshape(im.shape[1] * im.shape[2], 3)
    im2 = pca.fit_transform(im2)
    if pix is None:
        pix = im2
    else:
        pix = np.vstack([pix, im2])

# p = pca.fit_transform(pix)
kmeans = cluster(n_clusters=3, random_state=0).fit(pix)

for j, im in enumerate(data_test):
    im = im['image'].squeeze()
    # im = cv2.imread(i)
    # im = cv2.resize(im, (224, 224))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2 = im.reshape(im.shape[1] * im.shape[2], 3)
    im2 = pca.fit_transform(im2)
    l = kmeans.predict(im2)
# l = kmeans.labels_
#     l = l.reshape(im.shape[1], im.shape[2])
    l = pca.inverse_transform(l)
    # plt.imshow(l)
    fig, ax = plt.subplots(1, 5)
    ax[0].imshow(pca.inverse_transform(im2))
    ax[1].imshow(l)
    ax[2].imshow(l == 0)
    ax[3].imshow(l == 1)
    ax[4].imshow(l == 2)
    plt.show()

    if j == 5:
        break
    # break