import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import glob
import os
import json

dir1 = "/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/Chris - First 20 Images UCSF TF+/Chris - First 20 Images (UCSF TF+)/UCSF TF positive"
dir2 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/Lindsay - First 20 Images UCSF TF+ 2.tar_.download/Lindsay - First 20 Images (UCSF TF+) (#2).tar.download/Lindsay - First 20 Images (UCSF TF+) (#2)/UCSF TF positive'

dirs = [dir1, dir2]
sizes = []
for d in dirs:

    data_dir = d + '/ann/'
    image_dir = d + '/img/'

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
            # print(img.size)
            sizes.append(img.size)

            if img.size == (1024, 680):
                radius = 10
            elif img.size == (2240, 1488):
                radius = 15
            elif img.size == (3008, 2000):
                radius = 20
            elif img.size == (3872, 2592):
                radius = 25
            elif img.size == (4288, 2848):
                radius = 30
            else:
                print('Unknown image size: ' + img.size)

            draw = ImageDraw.Draw(img)
            for point in img_data['objects']:
                if point['classTitle'] == 'Grading Area':
                    coords = point['points']['exterior']
                    coords = [(p[0], p[1]) for p in coords]
                    draw.polygon(coords, outline='black')
                else:
                    if point['classTitle'] == 'Definite Follicle':
                        color = 'green'
                    elif point['classTitle'] == 'Ambiguous Follicle':
                        color = 'blue'
                    else:
                        print(point['classTitle'])
                    # print(point['points']['exterior'])
                    coords = point['points']['exterior'][0]
                    # draw.point(, fill='black')
                    draw.ellipse((coords[0] - radius, coords[1] - radius, coords[0] + radius, coords[1] + radius), fill=color)
            # plt.figure()
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(Image.open(img_file))
            ax[1].imshow(img)
            ax[0].set_title(img.size)
            ax[1].set_title(radius)
            # plt.imshow(img)

u, c = np.unique(np.array(sizes), axis=0, return_counts=True)
print(u, c)

plt.show()
            # break

