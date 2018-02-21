import os
import numpy as np 
from PIL import Image
import skimage.data
import skimage.transform
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt

def load_data(data_dir):
	"""
	Parameters:
	path of data dir
	Returns:
	images: np.float32
	labels: np.int64
	"""

	# Get all subdirectories of data_dir
	directories = [d for d in os.listdir(data_dir)
						if os.path.isdir(os.path.join(data_dir, d))]

	images = []
	labels = []
	
	# Loop all directories and get label and images
	for d in directories:
		d_dir = os.path.join(data_dir, d)

		# Get all path of file in d dir
		image_dires = [os.path.join(d_dir, f)
						for f in os.listdir(d_dir) if f.endswith(".png")]
		# Load file to images and labels
		for f in image_dires:
			image = Image.open(f)
			image = np.array(image)
			images.append(image)
			labels.append(int(d))

	images = np.array(images)
	labels = np.array(labels)

	return np.float32(images), labels

def preprocess(path_image):
	image = skimage.data.imread(path_image)
	standard_image = skimage.transform.resize(image, (28, 28))
	standard_image = rgb2gray(standard_image)
	standard_image = 1 - standard_image

	io.imsave('/home/kyobi/Desktop/nhandang/chuviet/data/testing/rand/3n.png', standard_image)
	print (standard_image)

	plt.imshow(standard_image)
	plt.show()


if __name__ == "__main__":

	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(dir_path, "../data/testing")
	data_dir = "/home/kyobi/Desktop/nhandang/chuviet/data/testing" 
	images, labels = load_data(data_dir)

	print (type(labels[0]))

	preprocess("/home/kyobi/Desktop/nhandang/chuviet/data/testing/rand/3.jpg")



