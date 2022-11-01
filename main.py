import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 


def greyscale(image):
    """ Converts pixel values of target image to greyscale by means of quadrature.
    """
    r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
    greyImg = np.sqrt(r.astype(float)**2 + g.astype(float)**2 + b.astype(float)**2)
    greyImg = greyImg/np.max(greyImg)

    return greyImg


