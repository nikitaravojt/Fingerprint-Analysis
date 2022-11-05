import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 


image_dir = 'images/'

def check_channels(img):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # cv version, value range is (0,255) instead of (0,1)

    return img


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


def binarize(image, thresh=127, adaptive=False):
    """Basic image thresholding (binarization) using cv2.threshold 
    """
    _, thresholded_img = cv.threshold(src=image.astype(np.uint8), thresh=thresh, maxval=255, type=cv.THRESH_BINARY)

    if adaptive:
        image = image.astype(np.uint8)
        thresholded_img = cv.adaptiveThreshold(src=image, maxValue=255, \
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv.THRESH_BINARY, \
            blockSize=9, C=1)
    
    return thresholded_img


def closing(img, size=3):
    """Performs closing operation on a binarized image input. A closing
    operation is comprised of dilation followed by erosion, with the
    aim of removing holes and missing pixels in ridge features. The input
    is inverted such that dilation and erosion is performed on the ridges
    and not the background. 

    Args:
        img (arr): binarized image
        size (int, optional): Size of kernel. Defaults to 3.

    Returns:
        arr: Closed image (dilated then eroded)
    """
    img = np.invert(img)
    centre = int(np.ceil(size/2) - 1)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (size, size),
                                    (centre, centre))

    dilated_img = cv.dilate(img, element)
    eroded_image = cv.erode(dilated_img, element)
    final = np.invert(eroded_image)

    return final


def opening(img, size=3):
    """Performs opening operation on a binarized image input. An opening
    operation is comprised of erosion followed by dilation, with the
    aim of removing noise from the image. The input is inverted such 
    that dilation and erosion is performed on the ridges and not the 
    background. 

    Args:
        img (arr): binarized image
        size (int, optional): Size of kernel. Defaults to 3.

    Returns:
        arr: Opened image (eroded then dilated)
    """
    img = np.invert(img)
    centre = int(np.ceil(size/2) - 1)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (size, size),
                                    (centre, centre))

    eroded_image = cv.erode(img, element)
    dilated_img = cv.dilate(eroded_image, element)
    final = np.invert(dilated_img)

    return final


def smoothing(img):
    """Performs smoothing operation on a binarized image by first
    opening then closing. This has the combined effect of de-noising 
    the image and then removing holes from ridge features.

    Args:
        img (arr): binarized image

    Returns:
        arr: smoothed image (opened then closed)
    """
    final = closing(opening(img))

    return final




img1 = plt.imread(image_dir+'set1_2.tif')
img1_gray = check_channels(img1)

img1_normalised = normalisation(img1_gray)

img1_segmented = segmentation(img1_normalised, width=12)

img1_binary = binarize(img1_normalised, thresh=151)
img1_adaptive = binarize(img1_normalised, adaptive=True)

# img1_smoothed = smoothing(img1_adaptive)

img1_open = opening(img1_adaptive)



# Plotting operations
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(331)
ax_gray = fig.add_subplot(332)
ax_norm = fig.add_subplot(333)
ax_segment = fig.add_subplot(334)
ax_glob_thresh = fig.add_subplot(335)
ax_adaptive_thresh = fig.add_subplot(336)
ax_dilated = fig.add_subplot(337)

ax.axis('off')
ax.set_title('Original')

ax_gray.axis('off')
ax_gray.set_title('Greyscale')

ax_norm.axis('off')
ax_norm.set_title('Normalised')

ax_segment.axis('off')
ax_segment.set_title('Segemented')

ax_glob_thresh.axis('off')
ax_glob_thresh.set_title('Global Thresholding', fontsize=10)

ax_adaptive_thresh.axis('off')
ax_adaptive_thresh.set_title('Adaptive Thresholding \n (Gaussian Method)', fontsize=10)

ax_dilated.axis('off')
ax_dilated.set_title('Dilated')

ax.imshow(img1, cmap='gray')
ax_gray.imshow(img1_gray, cmap='gray')
ax_norm.imshow(img1_normalised, cmap='gray')
ax_segment.imshow(img1_segmented, cmap='gray')
ax_glob_thresh.imshow(img1_binary, cmap='gray')
ax_adaptive_thresh.imshow(img1_adaptive, cmap='gray')
ax_dilated.imshow(img1_open, cmap='gray')

plt.show()