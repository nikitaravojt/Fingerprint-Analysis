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
    
    desired_mean = 150  
    desired_variance = 50

    normalised_image = np.where(image > mean, (desired_mean + np.sqrt(desired_variance*(image-mean)**2 / variance)), \
                                (desired_mean - np.sqrt(desired_variance*(image-mean)**2 / variance)))

    return normalised_image



def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))




def segmentation(image, width=24):

    # block_array = []
    # for row in range(0, image.shape[0], width):
    #     for col in range(0, image.shape[1], width):
    #         block_array.append(image[row:row + width, col:col + width]) 

    block_array = blockshaped(image, width, width)

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
        if (mean_array[index] > relative_mean) and (variance_array[index] < relative_var):
            block_array[index] = block_array[index] * 0

    segemented_image = unblockshaped(block_array, image.shape[0], image.shape[1])

    return segemented_image


def orient(image, width=6):

    # Computing gradient
    grad_x_sq_array = np.zeros((image.shape[0], image.shape[1]))
    grad_y_sq_array = np.zeros((image.shape[0], image.shape[1]))

    for row in range(1, image.shape[0]-1):
        for col in range(1, image.shape[1]-1):
            grad_x = (image[row+1, col] - image[row-1, col]) / 2
            grad_y = (image[row, col+1] - image[row, col-1]) / 2
            # edgeless_image[row, col] = np.array([grad_x, grad_y])
            grad_x_sq_array[row, col] = grad_x**2 - grad_y**2
            grad_y_sq_array[row, col] = 2*grad_x*grad_y

    

    block_grad_mean = np.zeros((image.shape[0]/width, image.shape[1]/width))
    for row in range(0, grad_x_sq_array.shape[0], width):
        for col in range(0, grad_x_sq_array.shape[1], width):
            pass
           


    print('f')




img1 = plt.imread(image_dir+'012_3_1.tif')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # cv version, value range is (0,255) instead of (0,1)
# img1_gray = greyscale(img1)

img1_normalised = normalisation(img1_gray)

img1_segmented = segmentation(img1_normalised, width=24)

# orient(img1_gray)

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