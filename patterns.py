import numpy as np


def load_terminations():
    term_1 = np.array([[1,0,1],
                       [1,0,1],
                       [1,1,1]])
    term_2 = np.array([[1,1,1],
                       [1,0,0],
                       [1,1,1]])
    term_3 = np.array([[1,1,1],
                       [1,0,1],
                       [1,0,1]])
    term_4 = np.array([[1,1,1],
                       [0,0,1],
                       [1,1,1]])
    term_5 = np.array([[1,1,0],
                       [1,0,1],
                       [1,1,1]])
    term_6 = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,0]])                       
    term_7 = np.array([[1,1,1],
                       [1,0,1],
                       [0,1,1]])
    term_8 = np.array([[0,1,1],
                       [1,0,1],
                       [1,1,1]])
    term_stack = np.stack((term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8))

    return term_stack


def load_bifurcations():
    bif_1 = np.array([[0,1,0],
                      [1,0,1],
                      [1,0,1]])
    bif_2 = np.array([[1,0,1],
                      [1,0,1],
                      [0,1,0]])
    bif_3 = np.array([[0,1,1],
                      [1,0,0],
                      [0,1,1]])                   
    bif_4 = np.array([[1,1,0],
                      [0,0,1],
                      [1,1,0]])
    bif_5 = np.array([[1,0,1],
                      [0,0,0],
                      [1,1,1]])
    bif_6 = np.array([[1,1,1],
                      [0,0,0],
                      [1,0,1]])
    bif_7 = np.array([[1,0,1],
                      [0,0,1],
                      [1,0,1]])
    bif_8 = np.array([[1,0,1],
                      [1,0,0],
                      [1,0,1]])
    bif_9 = np.array([[0,1,0],
                      [1,0,1],
                      [0,1,1]])
    bif_10 = np.array([[0,1,1],
                       [1,0,1],
                       [0,1,0]])
    bif_11 = np.array([[0,1,0],
                       [1,0,1],
                       [1,1,0]])
    bif_12 = np.array([[1,1,0],
                       [1,0,1],
                       [0,1,0]])
    bif_13 = np.array([[1,0,1],
                       [0,0,0],
                       [0,1,1]])
    bif_14 = np.array([[1,0,1],
                       [0,0,0],
                       [1,1,0]])
    bif_15 = np.array([[0,1,1],
                       [0,0,0],
                       [1,0,1]])
    bif_16 = np.array([[1,1,0],
                       [0,0,0],
                       [1,0,1]])
    bif_17 = np.array([[1,1,0],
                       [0,0,1],
                       [1,0,1]])
    bif_18 = np.array([[1,0,1],
                       [1,0,0],
                       [0,1,1]])
    bif_19 = np.array([[1,0,1],
                       [0,0,1],
                       [1,1,0]])
    bif_20 = np.array([[0,1,1],
                       [1,0,0],
                       [1,0,1]])
    bif_21 = np.array([[1,0,1],
                       [1,0,0],
                       [0,0,1]])
    bif_22 = np.array([[0,0,1],
                       [1,0,0],
                       [1,0,1]])
    bif_23 = np.array([[1,0,0],
                       [0,0,1],
                       [1,0,1]])
    bif_24 = np.array([[1,0,1],
                       [0,0,1],
                       [1,0,0]])

    bif_stack = np.stack((bif_1, bif_2, bif_3, bif_4, bif_5, bif_6, bif_7, bif_8, bif_9, bif_10,  
        bif_11, bif_12, bif_13, bif_14, bif_15, bif_16, bif_17, bif_18, bif_19, bif_20, bif_21,
            bif_22, bif_23, bif_24))

    return bif_stack

