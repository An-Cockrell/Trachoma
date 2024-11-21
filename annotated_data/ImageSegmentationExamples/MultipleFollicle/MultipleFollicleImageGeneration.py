import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import glob
import os
import json
import cv2 as cv
from scipy import spatial


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


# p = {'c': 0.051131867811012005, 'maxThresh': 142.98342462367424, 'minCirc': 0.3756499172238393, 'minConv': 0.8785338014729913, 'minThresh': 12.721471375830971}
p = {'maxThresh': 200, 'minThresh': 10, 'minCirc': 0.1, 'minConv': .75}

dir1 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Chris'
dir2 = '/media/dsocia22/T7/Trachoma/annotated_data/data from Supervisely task/MoreImages/UCSF TF positive Lindsay'

image_dir = '/media/dsocia22/T7/Trachoma/m/'

dirs = [dir1, dir2]
sizes = []
params = cv.SimpleBlobDetector_Params()

# # Change thresholds
params.minThreshold = p['minThresh']
params.maxThreshold = p['maxThresh']

# Filter by Area.
params.filterByArea = True
params.minArea = 75
params.maxArea = 5000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = p['minCirc']

# # # Filter by Convexity
params.filterByConvexity = True
params.minConvexity = p['minConv']

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

#filter by color
# params.filterByColor = True
# params.blobColor = p['c']
detector = cv.SimpleBlobDetector_create(params)
fe = FollicleEnhance()

num_mask = 0
for p, df in enumerate(dirs):
    if p == 0:
        name = 'chris'
    else:
        name = 'lindsay'
    img_num_save = 0
    data_dir = df + '/ann/'
    # image_dir = d + '/img/'

    data_files = glob.glob(data_dir + '*.json')
    # image_files = glob.glob(image_dir + '/*.jpg')
    # radius = 20

    for data_file in data_files:
        size = os.path.getsize(data_file)
        annot_foll = []
        definate_foll = []
        key_foll = []
        if size > 1000:
            # print(data_file.split('/')[-1][:-5])
            im_num = data_file.split('/')[-1][:-5]
            img_file = image_dir + im_num
            with open(data_file) as d:
                img_data = json.load(d)

            img = Image.open(img_file)
            # img2 = Image.open(img_file)
            # img3 = Image.open(img_file)
            # # print(img.size)
            # sizes.append(img.size)

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
            poly = Image.new('1', img.size)
            pdraw = ImageDraw.Draw(poly)
            for point in img_data['objects']:
                if point['classTitle'] == 'Grading Area':
                    coords = point['points']['exterior']
                    coords = [(p[0], p[1]) for p in coords]
                    pdraw.polygon(coords, fill=255)
                else:
                    if point['classTitle'] == 'Definite Follicle':
                        color = 'green'
                    elif point['classTitle'] == 'Ambiguous Follicle':
                        color = 'blue'
                    else:
                        print(point['classTitle'])
                    # print(point['points']['exterior'])
                    coords = point['points']['exterior'][0]
                    annot_foll.append(coords)
                    if color == 'green':
                        definate_foll.append(coords)
                    # draw.point(, fill='black')
                    # draw.ellipse((coords[0] - radius, coords[1] - radius, coords[0] + radius, coords[1] + radius),
                    # fill='black')
            inverted_poly = ImageOps.invert(poly)
            img.paste(poly, mask=inverted_poly.convert('1'))
            # img.putalpha(poly)/
            # img.show()
                #### BLOB DETECTION ####
            im_cv = cv.imread(img_file)
            im_cv = fe(im_cv)

            # Detect blobs.
            keypoints = detector.detect(im_cv)
            # print(len(keypoints))
            # Draw detected blobs as red circles.\
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob\
            # im_with_keypoints = cv.drawKeypoints(img2, keypoints, np.array([]), (255, 0, 0),
            #                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # draw = ImageDraw.Draw(img)
            kp_u = 0
            if len(keypoints) > 0:
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
                        # print(d)
                        # key = keypoints[pt]

                        # print(r)
                        if d < 30:
                            avg_r.append(r)
                            kp_u += 1

                avg_r = np.mean(avg_r)

                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(img)
                # ax[1].imshow(img2)
                # # plt.imshow(img)
                # plt.show()

                # print(avg_r)
                num_mask = num_mask + len(annot_foll)
                mask_ind = 0
                # for coord in annot_foll:
                x_shift = np.random.rand()
                y_shift = np.random.rand()
                # crop_img = img.crop((int(coord[0] - avg_r - (2 * avg_r * x_shift)),
                #                      int(coord[1] - avg_r - (2 * avg_r * y_shift)),
                #                      int(coord[0] + avg_r + (2 * avg_r * (1 - x_shift))),
                #                      int(coord[1] + avg_r + (2 * avg_r * (1 - y_shift)))))

                mask = Image.new('1', img.size)
                draw = ImageDraw.Draw(mask)
                for coord in annot_foll:
                    draw.ellipse((coord[0] - avg_r, coord[1] - avg_r, coord[0] + avg_r, coord[1] + avg_r), fill=color)
                # mask = mask.crop((int(coord[0] - avg_r - (2 * avg_r * x_shift)),
                #                   int(coord[1] - avg_r - (2 * avg_r * y_shift)),
                #                   int(coord[0] + avg_r + (2 * avg_r * (1 - x_shift))),
                #                   int(coord[1] + avg_r + (2 * avg_r * (1 - y_shift)))))

                img.save(
                    '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Images/{}_{}_{}.jpg'.format(
                        mask_ind, im_num.split('.')[0], name))
                mask.save(
                    '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Masks/{}_{}_{}.jpg'.format(
                        mask_ind, im_num.split('.')[0], name))
                mask_ind += 1
                print(img_num_save, mask_ind, num_mask, img.size)
            img_num_save += 1

images = glob.glob('/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Images/' + '*.jpg')
c_img = [im.split('/')[-1][:-10] for im in images if 'chris' in im]
l_im = [im.split('/')[-1][:-12] for im in images if 'lindsay' in im]

similar = list(set(c_img) & set(l_im))

test_ind = np.random.choice(np.arange(len(similar)), int(len(similar) * .1), replace=False).tolist()
test = [similar[i] for i in test_ind]
#
# c_test = []
# l_test = []

for t in test:
    c = t + '_chris.jpg'
    l = t + '_lindsay.jpg'

    os.replace('/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Images/' + c, '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Test/Chris/Images/' + c)
    os.replace('/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Images/' + l,
              '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Test/Lindsay/Images/' + l)
    os.replace('/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Masks/' + c,
              '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Test/Chris/Masks/' + c)
    os.replace('/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Masks/' + l,
              '/media/dsocia22/T7/Trachoma/annotated_data/FollicleDetection/MultipleFollicleImages/Test/Lindsay/Masks/' + l)
