import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import glob
import os
import json

dir1 = "/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/Chris - First 20 Images UCSF TF+/Chris - First 20 Images (UCSF TF+)/UCSF TF positive"
dir2 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/Lindsay - First 20 Images UCSF TF+ 2.tar_.download/Lindsay - First 20 Images (UCSF TF+) (#2).tar.download/Lindsay - First 20 Images (UCSF TF+) (#2)/UCSF TF positive'

image_dir = '/media/dsocia22/T7/Trachoma/m/'
dirs = [dir1, dir2]
sizes = []
ind = 0
saved_image = []
for d in dirs:

    data_dir = d + '/ann/'
    # image_dir = d + '/img/'

    data_files = glob.glob(data_dir + '*.json')
    # image_files = glob.glob(image_dir + '/*.jpg')
    # radius = 20

    for data_file in data_files:
        size = os.path.getsize(data_file)
        if size > 1000:
            print(data_file.split('/')[-1][:-5])
            img_file = image_dir + data_file.split('/')[-1][:-5]
            with open(data_file) as d:
                img_data = json.load(d)

            img = Image.open(img_file)

            mask = Image.new('1', img.size)
            draw = ImageDraw.Draw(mask)
            for point in img_data['objects']:
                if point['classTitle'] == 'Grading Area':
                    coords = point['points']['exterior']
                    coords = [(p[0], p[1]) for p in coords]
                    print(len(coords))
                    if len(coords) > 0:
                        draw.polygon(coords, fill='white')

                        # img.save('GradableAreaData/Images/{}.jpg'.format(ind))
                        # mask.save('GradableAreaData/Mask/{}.jpg'.format(ind))
                        ind += 1
                        saved_image.append(img_file)


dir3 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Chris'
dir4 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Lindsay'

dirs = [dir3, dir4]
sizes = []
ind = 0
saved_image_2 = []
for df in dirs:

    data_dir = df + '/ann/'
    # image_dir = d + '/img/'

    data_files = glob.glob(data_dir + '*.json')
    # image_files = glob.glob(image_dir + '/*.jpg')
    # radius = 20

    for data_file in data_files:
        size = os.path.getsize(data_file)
        if size > 1000:
            print(data_file.split('/')[-1][:-5])
            im_num = data_file.split('/')[-1][:-5]
            img_file = image_dir + im_num
            if img_file not in saved_image:
                with open(data_file) as d:
                    img_data = json.load(d)

                img = Image.open(img_file)

                mask = Image.new('1', img.size)
                draw = ImageDraw.Draw(mask)
                for point in img_data['objects']:
                    if point['classTitle'] == 'Grading Area':
                        coords = point['points']['exterior']
                        coords = [(p[0], p[1]) for p in coords]
                        print(len(coords))
                        if len(coords) > 0:
                            draw.polygon(coords, fill='white')

                            name = df.split(' ')[-1]
                            im_num = im_num[:-4]
                            if name == 'Chris':
                                img.save('GradableAreaData/Not_yet_used_in_training/Chris/Images/{}_{}.jpg'.format(im_num, name))
                                mask.save('GradableAreaData/Not_yet_used_in_training/Chris/Mask/{}_{}.jpg'.format(im_num, name))
                            else:
                                img.save(
                                    'GradableAreaData/Not_yet_used_in_training/Lindsay/Images/{}_{}.jpg'.format(im_num,
                                                                                                              name))
                                mask.save(
                                    'GradableAreaData/Not_yet_used_in_training/Lindsay/Mask/{}_{}.jpg'.format(im_num,
                                                                                                            name))
                            ind += 1
                            saved_image_2.append(img_file)


            # m = np.array(mask)
            # plt.figure()
#             fig, ax = plt.subplots(1, 2)
#             ax[0].imshow(img)
#             ax[1].imshow(mask)
#             ax[0].set_title(img.size)
#             # plt.imshow(img)
#
#             break
#
# u, c = np.unique(np.array(sizes), axis=0, return_counts=True)
# print(u, c)
#
# plt.show()