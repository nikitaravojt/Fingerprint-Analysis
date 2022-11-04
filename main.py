import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 


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


def block_stack(arr, block_height, block_width):
    """
    Strips image array into blocks, storing the blocks in a stack.
    Returns an array of blocks shaped (n, block_height, block_width).
    """
    arr_height, _ = arr.shape
    return (arr.reshape(arr_height//block_height, block_height, -1, block_width)
               .swapaxes(1,2)
               .reshape(-1, block_height, block_width))


def unblockify(arr, image_height, image_width):
    """
    Reshapes a block stack (from block_stack()) of shape 
    (n, block_height, block_width) into a 2D array of
    shape (image_height, image_width)
    """
    _, nrows, ncols = arr.shape
    return (arr.reshape(image_height//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(image_height, image_width))


def segmentation(image, width=24):

    block_array = block_stack(image, width, width)

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

    segemented_image = unblockify(block_array, image.shape[0], image.shape[1])

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