import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
import skimage.morphology as morph
from patterns import load_terminations, load_bifurcations


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
        if (mean_array[index] > global_block_mean):
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
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
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
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                    (centre, centre))

    eroded_image = cv.erode(img, element)
    dilated_img = cv.dilate(eroded_image, element)
    final = np.invert(dilated_img)

    return final


def smoothing(img, size=3):
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


def remove_holes(img):
    img = morph.remove_small_holes(img, area_threshold=15, connectivity=5)

    return img
    

def skeletonise(img):
    img_inv = np.invert(img.astype(bool)) 
    img_skeleton = morph.skeletonize(img_inv)
    img_skeleton = np.invert(img_skeleton)
    
    return img_skeleton


def locate_features(img, term_thresh=8, bif_thresh=6):
    """Identifies the row, col locations of all fingerprint ridge terminations 
    and bifurcations in the fingerprint image, based on pre-defined patterns.

    Args:
        img (arr): Skeletonised (thinned) fingerprint image

    Returns:
        arr: Coordinates of all ridge terminations
    """

    img = img.astype(int)

    term_stack = load_terminations() # loading terminations pattern stack
    bif_stack = load_bifurcations() # loading bifurcations pattern stack

    terminations_img = np.copy(img) # termination pixels will be marked with (int) 2
    bifurcations_img = np.copy(img) # bifurcations pixels will be marked with (int) 2
    black_coords = np.argwhere(img == 0) # identifying all black pixels in target image

    for coord in black_coords:  
        row, col = coord[0], coord[1]
        if (row != 0) and (row != img.shape[0]-1) and (col != 0) and (col != img.shape[1]-1):
            neighbourhood = [[img[row-1][col-1], img[row-1][col], img[row-1][col+1]], \
                            [img[row][col-1], img[row][col], img[row][col+1]], \
                            [img[row+1][col-1], img[row+1][col], img[row+1][col+1]]]

            for pattern in term_stack:
                if np.array_equal(neighbourhood, pattern):
                    terminations_img[row, col] = 2
            for pattern in bif_stack:
                if np.array_equal(neighbourhood, pattern):
                    bifurcations_img[row, col] = 2
              
    term_coords = np.argwhere(terminations_img == 2)
    bif_coords = np.argwhere(bifurcations_img == 2)
    

    # Checking for false positives
    term_coords_corrected = np.copy(term_coords)
    bif_coords_corrected = np.copy(bif_coords)
    target = np.array([-1, -1]) # setting up false positive target to remove later

    for index, coord in enumerate(term_coords):
        distances = np.sqrt((term_coords[:,0]-coord[0])**2 + (term_coords[:,1]-coord[1])**2)
        distances[index] = 999 # setting itself to high value to avoid removing itself
        removal_targets = np.where(distances < term_thresh)
        for removal_index in removal_targets:
            term_coords_corrected[removal_index] = -1.0
    
    term_coords_corrected = np.delete(term_coords_corrected, np.where(term_coords_corrected == target), axis=0)

    for index, coord in enumerate(bif_coords):
        distances = np.sqrt((bif_coords[:,0]-coord[0])**2 + (bif_coords[:,1]-coord[1])**2)
        distances[index] = 999 # setting itself to high value to avoid removing itself
        removal_targets = np.where(distances < bif_thresh)
        for removal_index in removal_targets:
            bif_coords_corrected[removal_index] = -1.0

    bif_coords_corrected = np.delete(bif_coords_corrected, np.where(bif_coords_corrected == target), axis=0)


    return (term_coords_corrected, bif_coords_corrected)


def highlight_features(img, term_coords, bif_coords, target_axis):
    """Outputs fingerprint image overlayed by a scatter plot of
    the locations of ridge terminations (endings).

    Args:
        img (arr): Image to overlay
        feature_coords (arr): Array containing the row,col positions of ridge terminations
        target_axis (plt object): matplotlib axis to plot over
    """
    target_axis.set_aspect('equal')
    target_axis.imshow(img, cmap='gray')
    target_axis.scatter(term_coords[:,1], term_coords[:,0], facecolors='none', edgecolors='b', label='Terminations')
    target_axis.scatter(bif_coords[:,1], bif_coords[:,0], facecolors='none', edgecolors='r', label='Bifurcations')
    target_axis.axis('off')
    target_axis.legend(loc='upper right')

    plt.show()


def plot_thresh_vs_features(img_skeleton, target_axis):
    thresholds = np.arange(0,16) # non-inclusive
    detected_features = np.zeros(len(thresholds))

    for index, thresh in enumerate(thresholds):
        term_locs, bif_locs = locate_features(img_skeleton, thresh, thresh)
        detected_features[index] = len(term_locs) + len(bif_locs)

    # Plotting operations
    target_axis.plot(thresholds, detected_features, '-.')
    target_axis.set_xlabel('Threshold Value')
    target_axis.set_ylabel('Detected Features (Total)')

    plt.savefig('Features_vs_Thresh.jpg')
    plt.show()




img1 = plt.imread(image_dir+'set1_1.tif')
img1_gray = check_channels(img1)

img1_normalised = normalisation(img1_gray)

img1_segmented = segmentation(img1_normalised, width=20)

img1_binary = binarize(img1_normalised, thresh=151)
img1_adaptive = binarize(img1_normalised, adaptive=True)

img1_smoothed = smoothing(img1_adaptive, size=25)

img1_skeletonised = skeletonise(img1_smoothed)


# Plotting operations
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(331)
ax_gray = fig.add_subplot(332)
ax_norm = fig.add_subplot(333)
ax_segment = fig.add_subplot(334)
ax_glob_thresh = fig.add_subplot(335)
ax_adaptive_thresh = fig.add_subplot(336)
ax_dilated = fig.add_subplot(337)
ax_skeleton = fig.add_subplot(338)

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
ax_dilated.set_title('Smoothed')

ax_skeleton.axis('off')
ax_skeleton.set_title('Skeletonised')

ax.imshow(img1, cmap='gray')
ax_gray.imshow(img1_gray, cmap='gray')
ax_norm.imshow(img1_normalised, cmap='gray')
ax_segment.imshow(img1_segmented, cmap='gray')
ax_glob_thresh.imshow(img1_binary, cmap='gray')
ax_adaptive_thresh.imshow(img1_adaptive, cmap='gray')
ax_dilated.imshow(img1_smoothed, cmap='gray')
ax_skeleton.imshow(img1_skeletonised, cmap='gray')

plt.show()



# Minutiae Extraction Output
fig2, ax2 = plt.subplots(1)
term_locations, bif_locations = locate_features(img1_skeletonised, term_thresh=8, bif_thresh=6)
highlight_features(img1_skeletonised, term_locations, bif_locations, ax2)

fig3, ax3 = plt.subplots(1)
plot_thresh_vs_features(img1_skeletonised, target_axis=ax3)
