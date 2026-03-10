from imutils import paths
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_blur_fft(image, size=60, thresh=25, vis=False):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	if vis:
		# compute the magnitude spectrum of the transform
		magnitude = 20 * np.log(np.abs(fftShift))
		# display the original input image
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		# display the magnitude image
		ax[1].imshow(magnitude, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		# show our plots
		plt.show()

	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return (mean, mean <= thresh)

def detect_bright(image, rgb_image, thresh=200, vis=False):
	blurred = cv2.GaussianBlur(image, (11, 11), 0)
	bright = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)[1]
	bright_count = np.sum(bright > 0)
	count = np.sum(bright == 255)
	if vis:
		# display the original input image
		(fig, ax) = plt.subplots(1, 1 )
		# ax[0].imshow(image, cmap="gray")
		ax.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
		ax.set_title("Input")
		ax.set_xticks([])
		ax.set_yticks([])

		(fig2, ax2) = plt.subplots(1, 1 )
		# display the magnitude image
		ax2.imshow(bright, cmap="gray")
		ax2.set_title('Glare Percentage: {:.4f}'.format((count/bright.size) * 100))
		ax2.set_xticks([])
		ax2.set_yticks([])

		# ax[1].imshow(rgb_image)
		# # ax[1].set_title('{:.4f}'.format(count / bright.size))
		# ax[1].set_xticks([])
		# ax[1].set_yticks([])
		# show our plots
		plt.show()

	return count


# img_dir = 'TrachomaData/tarsal plate zip/allTZphotos/allTZphotos'
# img_keys = 'TrachomaData/trachomagroundtruthkeyWungrades.csv'
#
# keys_df = pd.read_csv(img_keys, sep=',', header=0, usecols=['imagename', 'ans_ICAPS'])
img_dir = 'm'
img_keys = 'm/tfti.csv'
keys_df = pd.read_csv(img_keys, sep=',', header=0, usecols=['key'])


tresh = 10

non_blur = 0
blur = 0
min_blur = 100
max_blur = 0
ungrad_vals = []
grad_vals = []
ungrad_b = []
grad_b = []
# loop over the input images
for r in keys_df.iterrows():
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	# print(r[1]['key'])
	imagePath = os.path.join(img_dir, 'image' + str(r[1]['key'])) + '.jpg'
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# fm = variance_of_laplacian(gray)
	val, blurry = detect_blur_fft(gray, vis=False, thresh=15)
	count = detect_bright(gray, image, vis=True)
	# bright_per = count / gray.size
	text = "Not Blurry"
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if blurry:
		text = "Blurry"
		blur += 1
		if val < min_blur:
			min_blur = val

		# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		# # plt.title("{}: {:.2f}, Bright count: {}".format(text, val, count))
		# plt.title("{}: {:.2f}".format(text, val))
		# plt.axis('off')

	if not blurry:
		non_blur += 1
		if val > max_blur:
			max_blur = val

	# if r[1]['ans_ICAPS'] == 'Ungradeable':
	# 	ungrad_vals.append(val)
	# 	ungrad_b.append(bright_per)
	# else:
	grad_vals.append(val)
	# grad_b.append(bright_per)
	# show the image
	# cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	# cv2.imshow("Image", image)
	# key = cv2.waitKey(0)

	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	# plt.title("{}: {:.2f}, Bright count: {}".format(text, val, count))
	plt.title("{}: {:.2f}".format(text, val))
	plt.axis('off')
	plt.show()

print(non_blur, blur, min_blur, max_blur)


