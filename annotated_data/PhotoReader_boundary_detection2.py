import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import glob
import os
import json
import cv2 as cv
from scipy import spatial
from tqdm import tqdm
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


class FollicleEnhance(object):
    """Increases the contrast between the rest of the follicle and the eye"""

    def __init__(self, clipLimit=5.0, returnRGB=True, replace=None, sonly=False, addon=False):
        self.clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        self.rgb = returnRGB
        self.r = replace
        self.add = addon
        self.s=sonly

    def __call__(self, img):
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        cl = self.clahe.apply(img_hsv[:, :, 1])
        return cl

        # if self.s:
        #     img_s = np.repeat(cl[:, :, np.newaxis], 3, axis=2)
        #     return img_s
        # elif self.rgb:
        #     if self.r is None:
        #         if self.add:
        #             return np.dstack((img, cl))
        #         else:
        #             img_hsv[:, :, 1] = cl
        #             return cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
        #     else:
        #         img[:, :, self.r] = cl
        #         return img
        # else:
        #     img_hsv[:, :, 1] = cl
        #     return img_hsv


space = {'minThresh': hp.uniform('minThresh', 0, 200),
         'maxThresh': hp.uniform('maxThresh', 100, 255),
         'minCirc': hp.uniform('minCirc', 0, 1),
         'minConv': hp.uniform('minConv', 0, 1),
         'c': hp.uniform('c', 0, 255),
         'k': hp.choice('k', [0, 1, 2])}


# minTresh = np.linspace(0, 200, 6)
# maxTresh = np.linspace(100, 255, 6)
# minCirc = np.linspace(0, 1, 5)
# minConv = np.linspace(0, 1, 5)
# c = np.linspace(0, 255, 5)
#
# params2test = np.array(np.meshgrid(minTresh, maxTresh, minCirc, minConv, c)).T.reshape(-1, 5)
results = {}

def objective(p):
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # # Change thresholds
    params.minThreshold = p['minThresh']
    params.maxThreshold = p['maxThresh']

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 5000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = p['minCirc']

    # # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = p['minConv']

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    #filter by color
    params.filterByColor = True
    params.blobColor = p['c']

    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)

    dir1 = "/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/Chris - First 20 Images UCSF TF+/Chris - First 20 Images (UCSF TF+)/UCSF TF positive"
    dir2 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/Lindsay - First 20 Images UCSF TF+ 2.tar_.download/Lindsay - First 20 Images (UCSF TF+) (#2).tar.download/Lindsay - First 20 Images (UCSF TF+) (#2)/UCSF TF positive'

    dirs = [dir1, dir2]
    sizes = []
    fe = FollicleEnhance()

    reward = 10000

    for d in dirs:
        data_dir = d + '/ann/'
        image_dir = d + '/img/'

        data_files = glob.glob(data_dir + '*.json')
        # image_files = glob.glob(image_dir + '/*.jpg')
        # radius = 20

        for data_file in data_files:
            annot_foll = []
            key_foll = []
            size = os.path.getsize(data_file)
            if size > 1000:
                # print(data_file.split('/')[-1][:-5])
                img_file = image_dir + data_file.split('/')[-1][:-5]
                with open(data_file) as d:
                    img_data = json.load(d)

                img = Image.open(img_file)
                # img2 = Image.open(img_file)
                # img3 = Image.open(img_file)
                #
                # # print(img.size)
                # sizes.append(img.size)
                #
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
                #
                # draw = ImageDraw.Draw(img)
                for point in img_data['objects']:
                    if (point['classTitle'] == 'Definite Follicle') | (point['classTitle'] == 'Ambiguous Follicle'):
                        # coords = point['points']['exterior']
                        # coords = [(p[0], p[1]) for p in coords]
                    #     # draw.polygon(coords, outline='black')
                    # else:
                    #     if point['classTitle'] == 'Definite Follicle':
                    #         color = 'green'
                    #     elif point['classTitle'] == 'Ambiguous Follicle':
                    #         color = 'blue'
                    #     else:
                    #         print(point['classTitle'])
                        # print(point['points']['exterior'])
                        coords = point['points']['exterior'][0]
                        annot_foll.append(coords)
                        # draw.point(, fill='black')
                        # draw.ellipse((coords[0] - radius, coords[1] - radius, coords[0] + radius, coords[1] + radius), fill=color)

                #### BLOB DETECTION ####
                if p['k'] == 0:
                    im_cv = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
                elif p['k'] == 1:
                    im_cv = cv.imread(img_file)
                    im_cv = fe(im_cv)
                elif p['k'] == 2:
                    im_cv = cv.imread(img_file)
                    im_cv = fe(im_cv)
                    rt, im_cv = cv.threshold(im_cv, 80, 255, cv.THRESH_BINARY)

                # Detect blobs.
                keypoints = detector.detect(im_cv)
                # print(len(keypoints))
                # Draw detected blobs as red circles.\
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob\
                # im_with_keypoints = cv.drawKeypoints(im_cv, keypoints, np.array([]), (255, 0, 0),
                #                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                if len(keypoints) > 0:
                    reward = 0
                    # draw = ImageDraw.Draw(img2)
                    for key in keypoints:
                        coords = key.pt
                        key_foll.append(coords)
                    key_foll = np.array(key_foll)
                    annot_foll = np.array(annot_foll).squeeze()

                    dist = spatial.distance.cdist(key_foll, annot_foll)
                    closest_pts = np.argmin(dist, axis=0)
                    avg_r = []
                    for i, key in enumerate(keypoints):
                        coords = key.pt
                        r = key.size / 2
                        if i in closest_pts:
                            ind = np.where(i == closest_pts)[0][0]
                            d = dist[i, ind]
                            print(d)
                            # key = keypoints[pt]

                            # print(r)
                            # if d < 30:
                                # draw.ellipse((coords[0] - r, coords[1] - r, coords[0] + r, coords[1] + r),
                                #              fill='red')
                            avg_r.append(r)
                            reward += d
                            # else:
                                # draw.ellipse((coords[0] - r, coords[1] - r, coords[0] + r, coords[1] + r),
                                #              fill='blue')
                        # else:
                        #     draw.ellipse((coords[0] - r, coords[1] - r, coords[0] + r, coords[1] + r),
                        #                  fill='black')

                    # draw = ImageDraw.Draw(img3)
                # avg_r = np.mean(avg_r)
                # print("AVERAGE RADIUS: " + str(avg_r) + " Radius: " + str(radius))
                results[reward] = p
    print(p, reward)
    return reward

best = fmin(objective, space, algo=tpe.suggest, max_evals=10000)

print(best)
print(space_eval(space, best))
                # for point in img_data['objects']:
                #     if point['classTitle'] == 'Grading Area':
                #         coords = point['points']['exterior']
                #         coords = [(p[0], p[1]) for p in coords]
                #         draw.polygon(coords, outline='black')
                #     else:
                #         if point['classTitle'] == 'Definite Follicle':
                #             color = 'green'
                #         elif point['classTitle'] == 'Ambiguous Follicle':
                #             color = 'blue'
                #         else:
                #             print(point['classTitle'])
                #         # print(point['points']['exterior'])
                #         coords = point['points']['exterior'][0]
                #         # draw.point(, fill='black')
                #         draw.ellipse((coords[0] - avg_r, coords[1] - avg_r, coords[0] + avg_r, coords[1] + avg_r),
                #                      fill=color)


#             # plt.figure()
#             fig, ax = plt.subplots(1, 3)
#             ax[0].imshow(img)
#             ax[1].imshow(img2)
#             ax[2].imshow(img3)
#             ax[1].set_title(img.size)
#             ax[0].set_title(radius)
#             ax[2].set_title(avg_r)
#             # plt.imshow(img)
#
# u, c = np.unique(np.array(sizes), axis=0, return_counts=True)
# print(u, c)
#
# plt.show()
#             # break

#{'c': 0.051131867811012005, 'k': 1, 'maxThresh': 142.98342462367424, 'minCirc': 0.3756499172238393, 'minConv': 0.8785338014729913, 'minThresh': 12.721471375830971}