import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from skimage import draw
from PIL import Image
from tqdm import tqdm

# class SynthTrachoma:
#     def __init__(self):
eye_dir = 'Eyelids/*.jpg'
not_eye_dir = 'NotEyelids/*.jpg'


def rgb_cov(im):
    '''
    Assuming im a torch.Tensor of shape (H,W,3):
    '''
    im_re = im.reshape(-1, 3)
    im_re -= im_re.mean(0, keepdim=True)
    return 1/(im_re.shape[0]-1) * im_re.T @ im_re


def image_stats(img_dir):

    r = []
    g = []
    b = []
    
    images = glob.glob(img_dir)
    for im in images:
        im = cv2.imread(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        b.extend(im[:, :, 2].flatten().tolist())
        g.extend(im[:, :, 1].flatten().tolist())
        r.extend(im[:, :, 0].flatten().tolist())
    
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    # TODO: use covariance instead of std
    d = np.vstack((r, g, b))
    cov = np.cov(d)
    mu = np.mean(d, axis=1)
    # b_mu = b.mean()
    # b_std = b.std()
    # g_mu = g.mean()
    # g_std = g.std()
    # r_mu = r.mean()
    # r_std = r.std()
    
    # return {'b_mu': b_mu, 'b_std': b_std, 'r_mu': r_mu, 'r_std': r_std, 'g_mu': g_mu, 'g_std': g_std, 'cov': cov}
    return {'mu': mu, 'cov': cov}


def gen_im(stats, size=224):
#     r = np.random.normal(stats['r_mu'], stats['r_std'], [size, size]).astype(int)
#     g = np.random.normal(stats['g_mu'], stats['g_std'], [size, size]).astype(int)
#     b = np.random.normal(stats['b_mu'], stats['b_std'], [size, size]).astype(int)

    # r = r / np.max(r)
    # g = g / np.max(g)
    # b = b / np.max(b)

    # im_syn = np.stack([r, g, b], axis=2)
    im_syn = np.random.multivariate_normal(stats['mu'], stats['cov'], (size, size)).astype(int)

    im_syn = np.clip(im_syn, 0, 255)
    
    return im_syn


def gen_eye(TF = True):
    x = np.random.randint(96, 136)
    y = np.random.randint(96, 136)
    x_radius = np.random.randint(156 / 2, 210 / 2)
    y_radius = np.random.randint(30, 60)
    # c_radius = c / np.sqrt(1 - (r / r_radius) ** 2)
    rotation = np.deg2rad(np.random.normal(0, 5))

    rr, cc = draw.ellipse(x, y, y_radius, x_radius, shape=[224, 224], rotation=rotation)

    # generate follicles
    if TF:
        num = np.random.randint(5, 20)
    else:
        num = np.random.randint(0, 4)

    follicles = []

    for i in range(num):
        while True:
            radius = np.random.randint(2, 6)
            x_c = np.random.randint(x - x_radius, x + x_radius)
            y_c = np.random.randint(y - y_radius, y + y_radius)
            # print(x_c, y_c)
            # print((((np.cos(rotation) * (x_c - x) + np.sin(rotation) * (y_c - y)) ** 2) / (x_radius - radius) ** 2) + (((np.sin(rotation) * (x_c - x) - np.cos(rotation) * (y_c - y)) ** 2) / (y_radius - radius) ** 2))
            if ((((np.cos(rotation) * (x_c - x) + np.sin(rotation) * (y_c - y)) ** 2) / (y_radius - radius) ** 2) + (((np.sin(rotation) * (x_c - x) - np.cos(rotation) * (y_c - y)) ** 2) / (x_radius - radius) ** 2)) < 1:
                # print('works')
                break

        f = draw.disk((x_c, y_c), radius)
        follicles.append(f)

    return rr, cc, follicles


eye = image_stats(eye_dir)
Neye = image_stats(not_eye_dir)

# eye = {'b_mu': 75.64485597021566, 'b_std': 35.587378791190744, 'r_mu': 151.1293205776574, 'r_std': 37.93683467507247, 'g_mu': 81.90694097429615, 'g_std': 37.41177299900325}
# Neye = {'b_mu': 61.06953203096222, 'b_std': 46.87585099277787, 'r_mu': 90.04605089450689, 'r_std': 45.6920172307766, 'g_mu': 67.23968574759101, 'g_std': 45.761433924444574}

for i in tqdm(range(3000)):
    eye_im = gen_im(eye)
    neye_im = gen_im(Neye)

    eye_shape = gen_eye(TF=False)
    im_syn = neye_im
    im_syn[eye_shape[0], eye_shape[1]] = eye_im[eye_shape[0], eye_shape[1]]

    for f in eye_shape[2]:
        im_syn[f[0], f[1]] = 255

    im = Image.fromarray(im_syn.astype(np.uint8))
    im.save('synthetic_NonTF_images/{}_no_tf.jpg'.format(str(i)))
    # fig = plt.figure(i)
    # plt.imshow(im_syn)

# for i in range(2, 4):
#     eye_im = gen_im(eye)
#     neye_im = gen_im(Neye)
#
#     eye_shape = gen_eye(TF=False)
#     im_syn = neye_im
#     im_syn[eye_shape[0], eye_shape[1]] = eye_im[eye_shape[0], eye_shape[1]]
#
#     for f in eye_shape[2]:
#         im_syn[f[0], f[1]] = 255
#     fig = plt.figure(i)
#     plt.imshow(im_syn)




# for i in range(5):
#     r = np.random.normal(r_mu, r_std, [size, size]).astype(int)
#     g = np.random.normal(g_mu, g_std, [size, size]).astype(int)
#     b = np.random.normal(b_mu, b_std, [size, size]).astype(int)
#
#     # r = r / np.max(r)
#     # g = g / np.max(g)
#     # b = b / np.max(b)
#
#     im_syn = np.stack([r, g, b], axis=2)
#     im_syn = np.clip(im_syn, 0, 255)
#     fig = plt.figure(i)
#     plt.imshow(im_syn)
#     plt.show()