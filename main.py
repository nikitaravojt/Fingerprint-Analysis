import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
from skimage.color import rgb2gray

image_dir = 'images/'

def greyscale(image):
    """ Converts pixel values of target image to greyscale by means of quadrature.
    """
    r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
    greyImg = np.sqrt(r.astype(float)**2 + g.astype(float)**2 + b.astype(float)**2)
    greyImg = greyImg/np.max(greyImg)

    return greyImg


def normalisation(image):
    """_summary_

    Args:
        image (arr): 2D array containing the greyscale pixel values of target image
    """
    mean = np.sum(image) / np.size(image)
    variance = np.sum((image - mean)**2) / np.size(image)
    
    desired_mean = 150/255
    desired_variance = 50/255

    normalised_image = np.where(image > mean, (desired_mean + np.sqrt(desired_variance*(image-mean)**2 / variance)), \
                                (desired_mean - np.sqrt(desired_variance*(image-mean)**2 / variance)))

    return normalised_image



def segmentation(image):
    width = 24
    block_array = np.array([image[x:x + width,y:y + width] 
          for x in range(0,image.shape[0],width) 
          for y in range(0,image.shape[1],width)])

    mean_array = []
    variance_array = []
    for block in block_array:
        block_mean = np.mean(block)
        block_var = np.sum((block - block_mean)**2) / np.size(block)

        mean_array.append(block_mean)
        variance_array.append(block_var)

    global_block_mean = np.mean(mean_array)
    global_block_var = np.mean(variance_array)

    background_blocks1 = []
    background_blocks2 = []
    for index, block in enumerate(block_array):
        if (mean_array[index] < global_block_mean):
            background_blocks1.append(block)
    relative_mean = np.mean(background_blocks1)
    
    for index, block in enumerate(block_array):
        if (variance_array[index] < global_block_var):
            background_blocks2.append(block)
    relative_var = np.mean(background_blocks2)


    for index, block in enumerate(block_array):
        if (mean_array[index] < relative_mean) and (variance_array[index] < relative_var):
            block_array[index] = block_array[index] * 0

    segemented_image = np.reshape(block_array, (image.shape[0], image.shape[1]))

    return segemented_image



img1 = plt.imread(image_dir+'012_3_3.tif')
# img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # cv version, value range is (0,255) instead of (0,1)
img1_gray = greyscale(img1)

img1_normalised = normalisation(img1_gray)

img1_segmented = segmentation(img1_normalised)


# Plotting operations
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(221)
ax_gray = fig.add_subplot(222)
ax_norm = fig.add_subplot(223)
ax_segment = fig.add_subplot(224)
ax.axis('off')
ax.set_title('Original')
ax_gray.axis('off')
ax_gray.set_title('Greyscale')
ax_norm.axis('off')
ax_norm.set_title('Normalised')
ax_segment.axis('off')
ax_segment.set_title('Segemented')
ax.imshow(img1)
ax_gray.imshow(img1_gray, cmap='gray')
ax_norm.imshow(img1_normalised, cmap='gray')
ax_segment.imshow(img1_segmented, cmap='gray')
plt.show()