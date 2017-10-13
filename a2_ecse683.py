"""
:author: Mido Assran
:date: Oct. 12, 2017
"""

# BLAS computing
import numpy as np
# Computer Vision module
import cv2
# Plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def imshow(img, save_fname_img="default.jpg"):
    """ Helper to show image """

    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite(save_fname_img, img)
        cv2.destroyAllWindows()

def gamutshow(img, save_fname_img="default_gamut.jpg"):
    """ Plot the gamut of the image """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    for pix_row in img:
        for pix in pix_row:
            ax.scatter(pix[0], pix[1], pix[2], c='k')

    ax.set_xlabel('B Label')
    ax.set_ylabel('G Label')
    ax.set_zlabel('R Label')

    plt.show()

def grey_edge_cc(img, minkowski_ord=0.15, grad_ord=3, save_fname_img="default_greyedge.jpg"):
    """ Perform grey-edge color constancy """

    img = img.astype(np.float32)
    sobel = np.copy(img)
    for _ in range(grad_ord):
        sobel = cv2.Sobel(sobel, cv2.CV_64F, 1, 0, ksize=-1)

    sobel = np.abs(sobel)

    kai = [0, 0, 0]
    kai[0] = np.mean(sobel[:, :, 0]**minkowski_ord)**(1.0/float(minkowski_ord))
    kai[1] = np.mean(sobel[:, :, 1]**minkowski_ord)**(1.0/float(minkowski_ord))
    kai[2] = np.mean(sobel[:, :, 2]**minkowski_ord)**(1.0/float(minkowski_ord))

    avg = np.mean(kai)
    for chan in range(3):
        k_chan = avg / kai[chan]
        img[:, :, chan] *= k_chan
    img = np.minimum(img, 255.0)
    img = np.uint8(img)

    imshow(img, save_fname_img)

def grey_world_cc(img, save_fname_img="default_greyworld.jpg"):
    """ Perform grey-world color constancy """

    img = img.astype(np.float32)
    avg = img.mean()
    for chan in range(3):
        k_chan = avg / img[:, :, chan].mean()
        img[:, :, chan] *= k_chan
        img[:, :, chan] = np.minimum(img[:, :, chan], 255.0)
        
    print(img.max(), np.uint8(img.max()))
    img = img.astype(np.uint8)

    imshow(img, save_fname_img)

def max_rgb_cc(img, save_fname_img="default_maxrgb.jpg"):
    """ Perform max-rgb color constancy """

    img = img.astype(float)
    white_bgr = [255.0, 255.0, 255.0]
    for chan in range(3):
        k_chan = white_bgr[0] / img[:, :, chan].max()
        img[:, :, chan] *= k_chan
        img[:, :, chan] = np.minimum(img[:, :, chan], 255.0)

    img = img.astype(np.uint8)

    imshow(img, save_fname_img)

def main():
    """ Main method used to run a2 """

    fname_img = 'img_2.jpg'
    img = cv2.imread(fname_img, -1)
    imshow(img)
    # gamutshow(img)
    # max_rgb_cc(img)
    # grey_world_cc(img)
    grey_edge_cc(img, minkowski_ord=6, grad_ord=1)

if __name__ == "__main__":
    main()
