import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class TrachomaDataset(Dataset):
    def __init__(self, img_dir, img_keys, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.img_keys = img_keys

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.img_dir, '0' + self.img_keys[item, 0]) + '.jpg'
        image = io.imread(img_path)

        if self.transform:
            for t in self.transform:
                image = t(image)

        sample = {'image': image, 'label': self.img_keys[item, 1]}

        return sample


# Transformation classes
class ToTensor(object):
    """Converts numpy array to torch tensor"""

    def __call__(self, img):
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))/255
        return torch.from_numpy(img).float()


class RandomFlip(object):
    """Randomly flips the image vertically"""


def generate_data_tf_loaders(img_dir, img_keys_csv, img_col, key_col, over_sample=True, transform=None, batch_size=4, num_workers=0):
    """Splits the dataset into test and train and returns a dataloader
     can over sample unbalanced class"""

    labels = pd.read_csv(img_keys_csv, sep=',', header=0, usecols=[img_col, key_col])

    # changes labels from no TF and TF to 0 and 1 respectively
    label_map = {'No TF': 0, 'TF': 1}
    labels = labels.replace({key_col: label_map})
    labels = np.asarray(labels.values)

    # shuffle dataset
    np.random.shuffle(labels)

    # splits No TF and TF for equal distribution in training and testing and validation
    no_tf = labels[labels[:, 1] == 0]
    tf = labels[labels[:, 1] == 1]

    # test train
    train_no_tf_num = int(len(no_tf) * .8)
    train_tf_num = int(len(tf) * .8)

    train_no_tf = no_tf[:train_no_tf_num, :]
    train_tf = tf[:train_tf_num, :]

    test_no_tf = no_tf[train_no_tf_num:, :]
    test_tf = tf[train_tf_num:, :]

    # train validate
    train_no_tf_num = int(len(train_no_tf) * .8)
    train_tf_num = int(len(train_tf) * .8)

    val_no_tf = train_no_tf[train_no_tf_num:, :]
    val_tf = train_tf[train_tf_num:, :]

    train_no_tf = train_no_tf[:train_no_tf_num, :]
    train_tf = train_tf[:train_tf_num, :]

    # over sample TF images to improve the class distribution
    if over_sample:
        # number of new sample to generate  - want 20% tf class
        num_gen = int(train_no_tf_num * .2) - train_tf_num
        random_ind = np.random.choice(train_tf_num, num_gen)
        train_tf = np.vstack((train_tf, train_tf[random_ind]))

    # combind tf and no tf
    train = np.vstack((train_tf, train_no_tf))
    val = np.vstack((val_tf, val_no_tf))
    test = np.vstack((test_tf, test_no_tf))

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    # create datasets
    train = TrachomaDataset(img_dir, train, transform=transform)
    val = TrachomaDataset(img_dir, val, transform=transform)
    test = TrachomaDataset(img_dir, test, transform=transform)

    return {'train': DataLoader(train, batch_size=batch_size, num_workers=num_workers),
            'val': DataLoader(val, batch_size=batch_size, num_workers=num_workers),
            'test': DataLoader(test, batch_size=batch_size, num_workers=num_workers)}


if __name__ == '__main__':
    # img_dir = 'TrachomaData/allTZphotos' # missing photos
    img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos' # unzipped file package contains more photos than entries in csv
    img_keys = 'TrachomaData/trachomagroundtruthkey.csv'

    trans = [ToTensor(), transforms.Resize(256), transforms.CenterCrop(224)] #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = generate_data_tf_loaders(img_dir, img_keys, 'imagename', 'ans_ground', transform=trans, batch_size=10, over_sample=False)
#     loader = TrachomaDataset(img_dir, img_keys, 'imagename', 'ans_ground', transform=trans)
#
#     tf_img = sum(loader.img_keys[:, 1])
#
#     print('Number of No TF images: %f' % ((len(loader) - tf_img)/len(loader)))
#     print('Number of TF images: %f' % (tf_img/len(loader)))
#
#     plt.bar(['No TF', 'TF'], [(len(loader) - tf_img)/len(loader), tf_img/len(loader)])
#     plt.ylabel('Image Percentage')
#
#     # # check: compare number of images to number of labels
#     # print(len([name for name in os.listdir(img_dir)]))
#     # print(len(loader))
#     #
#     # for i, sample in enumerate(loader):
# #         plt.imshow(sample['image'].permute(1, 2, 0))
# #         plt.title(sample['label'])
# #         plt.show()
#     #     print(sample['image'].shape)
#     #     if i > 3:
#     #         break
#
#     dataloader = DataLoader(loader, batch_size=4, shuffle=True, num_workers=0)
#
    def show_batch(sample_batch):
        img_batch, labels = sample_batch['image'], sample_batch['label']
        batch_size = len(img_batch)

        im_size = img_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(img_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))


    mean = 0.
    std = 0.
    nb_samples = 0.
    for i_batch, sample_batched in enumerate(dataloader['train']):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'])
        #
        # # observe 4th batch and stop.
        # if i_batch == 0:
        #     plt.figure()
        #     show_batch(sample_batched)
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        #     break

        data = sample_batched['image']

        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

    print(mean)
    print(std)

    stats = {'mean': mean.tolist(), 'std': std.tolist()}

    with open('Trachoma_dataset_stats.json', 'w') as f:
        json.dump(stats, f)