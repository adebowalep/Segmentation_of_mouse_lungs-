import numpy as np
import nrrd
import matplotlib.pyplot as plt
#% matplotlib inline
import cv2  as cv
from skimage.segmentation import morphological_chan_vese,checkerboard_level_set, disk_level_set
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.segmentation import inverse_gaussian_gradient,morphological_geodesic_active_contour
from skimage.color import rgb2gray
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
from skimage.filters import threshold_otsu, rank,threshold_local
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_mean
from skimage.filters import threshold_multiotsu



def read_data(filename):
    data, header = nrrd.read(filename)
    return data, header

def display_data(data1, data2, data3, data4):
    f, axarr = plt.subplots(2,2,figsize=(10,10))
    axarr[0,0].imshow(data1, cmap='gray')
    axarr[0, 0].set_title(' Slide 100')
    axarr[0,1].imshow(data2, cmap='gray')
    axarr[0, 1].set_title('slide 200')
    axarr[1,0].imshow(data3, cmap='gray')
    axarr[1, 0].set_title('slide 300')
    axarr[1,1].imshow(data4, cmap='gray')
    axarr[1, 1].set_title('slide 400')
    plt.show()
    
def get_data(array):
    li = []
    for i in array.ravel():
        if (i <600) & (i>250):
            li.append(i)
    return li
        

def manual_segmentation(array):
    thresholds= [170, 255]
    segmentation=np.where((array>thresholds[0])&(array<thresholds[1]),0,1)
    return segmentation,thresholds
    
def display_manual(array):
    
    segmentation,thresholds =manual_segmentation(array)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))

    # Plotting the original image.
    ax[0].imshow(array, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    # Plotting the histogram and the manual threshold segmentation obtained from
    # multi-Otsu.
    ax[1].hist(get_data(array), bins=255)
    ax[1].set_title('Histogram')
    ax[1].set_xlabel("Grayscale value")
    ax[1].set_ylabel("Pixel count")
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(segmentation,cmap=plt.cm.gray)
    ax[2].set_title('Manual thresholding between the threshold  170 and 255')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()
    


def otsu_segmentation(array):
    #Otsu Thresholding
    otsu_thresh = threshold_otsu(array)
    otsu_seg = np.where((array > otsu_thresh),1,0).astype(np.bool) 
    
    return otsu_seg, otsu_thresh 
    
    
def dislay_otsu(array):
    
    otsu_seg,otsu_thresh =otsu_segmentation(array)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))

    # Plotting the original image.
    ax[0].imshow(array, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the manual threshold segmentation obtained from
    # multi-Otsu.
    ax[1].hist(get_data(array), bins=255)
    ax[1].set_title('Histogram')
    ax[1].set_xlabel("Grayscale value")
    ax[1].set_ylabel("Pixel count")
    ax[1].axvline(otsu_thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(otsu_seg,cmap=plt.cm.gray)
    ax[2].set_title('OTSU segmentation')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()


def otsu_filter(array):
    
    #filters
    median = cv.medianBlur(np.uint16(array),3)
    mean = cv.blur(np.uint16(array),(3,3))
    gauss = cv.GaussianBlur(np.uint16(array),(3,3),0)
    
    #thresholds
    median_otsu_thresh = threshold_otsu(median)
    mean_otsu_thresh = threshold_otsu(mean)
    gauss_otsu_thresh = threshold_otsu(gauss)
    
    #segmentations
    median_otsu_seg = np.where((array > median_otsu_thresh),0,1).astype(np.uint8) 
    mean_otsu_seg = np.where((array > mean_otsu_thresh),0,1).astype(np.uint8)
    gauss_otsu_seg = np.where((array > gauss_otsu_thresh),0,1).astype(np.uint8)
    
    return  median_otsu_seg,mean_otsu_seg,gauss_otsu_seg
    
def display_otsu_filter(array):
    
    otsu_segment,_ = otsu_segmentation(array)
    
    median_otsu_seg,mean_otsu_seg,gauss_otsu_seg = otsu_filter(array)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(15, 7))
    axes = axes.ravel()

    axes[0].imshow(otsu_segment, cmap=plt.cm.gray)
    axes[0].set_title('Otsu image before filtering ')

    axes[1].imshow(gauss_otsu_seg, cmap=plt.cm.gray)
    axes[1].set_title('Gaussian filter')

    axes[2].imshow(mean_otsu_seg, cmap=plt.cm.gray)
    axes[2].set_title('Mean filter')

    axes[3].imshow(median_otsu_seg, cmap=plt.cm.gray)
    axes[3].set_title('Median filter')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def local_segmentation(array):
    #LOCAL THRESHOLD
    block_size = 35
    local_thresh = threshold_local(array, block_size, offset=50)
    local_seg = np.where((array > local_thresh),0,1).astype(np.uint8)
    return local_seg, local_thresh
    
def display_local(array):
    
    local_seg,local_thresh =local_segmentation(array)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))

    # Plotting the original image.
    ax[0].imshow(array, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the manual threshold segmentation obtained from
    # multi-Otsu.
    ax[1].hist(get_data(array), bins=255)
    ax[1].set_title('Histogram')
    ax[1].set_xlabel("Grayscale value")
    ax[1].set_ylabel("Pixel count")
    for thresh in local_thresh:
        ax[1].axvline(thresh.all(), color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(local_seg, cmap=plt.cm.gray)
    ax[2].set_title('Local segmentation with an off-set  value of 100')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()



def mean_segmentation(array):
    #Mean segmentation 
    mean_thresh = threshold_mean(array)
    mean_seg = np.where((array > mean_thresh),1,0).astype(np.uint8)
    return mean_seg,mean_thresh
    
def display_mean(array):
    
    mean_seg,mean_thresh =mean_segmentation(array)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))

    # Plotting the original image.
    ax[0].imshow(array, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the manual threshold segmentation obtained from
    # multi-Otsu.
    ax[1].hist(get_data(array), bins=255)
    ax[1].set_title('Histogram')
    ax[1].set_xlabel("Grayscale value")
    ax[1].set_ylabel("Pixel count")
    ax[1].axvline(mean_thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(mean_seg, cmap=plt.cm.gray)
    ax[2].set_title('Mean segmentation')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()


def mean_filter_segmentation(array):
    
    #filters
    median = cv.medianBlur(np.uint16(array),3)
    mean = cv.blur(np.uint16(array),(3,3))
    gauss = cv.GaussianBlur(np.uint16(array),(3,3),0)
    
    #thresholds
    median_otsu_thresh = threshold_mean(median)
    mean_otsu_thresh = threshold_mean(mean)
    gauss_otsu_thresh = threshold_mean(gauss)
    
    #segmentations
    median_seg = np.where((array > median_otsu_thresh),1,0).astype(np.uint8) 
    mean_seg = np.where((array > mean_otsu_thresh),1,0).astype(np.uint8)
    gauss_seg = np.where((array > gauss_otsu_thresh),1,0).astype(np.uint8)
    
    return  median_seg,mean_seg,gauss_seg
    
def display_mean_filter(array):
    
    mean_segment,_ = mean_segmentation(array)
    
    median_seg,mean_seg,gauss_seg = otsu_filter(array)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(15, 7))
    axes = axes.ravel()

    axes[0].imshow(mean_segment, cmap=plt.cm.gray)
    axes[0].set_title('Mean image before filtering ')

    axes[1].imshow(gauss_seg, cmap=plt.cm.gray)
    axes[1].set_title('Gaussian filter')

    axes[2].imshow(mean_seg, cmap=plt.cm.gray)
    axes[2].set_title('Mean filter')

    axes[3].imshow(median_seg, cmap=plt.cm.gray)
    axes[3].set_title('Median filter')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    

def multiotsu_segmentation(array):
    
    # Applying multi-Otsu threshold for the default value, generate three classes.
    thresholds = threshold_multiotsu(array)
    
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(array, bins=thresholds).astype('uint8')
    
    return regions,thresholds
    
def display_multiotsu(array):
    
    regions, thresholds =multiotsu_segmentation(array)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))

    # Plotting the original image.
    ax[0].imshow(array, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the manual threshold segmentation obtained from
    # multi-Otsu.
    ax[1].hist(get_data(array), bins=255)
    ax[1].set_title('Histogram')
    ax[1].set_xlabel("Grayscale value")
    ax[1].set_ylabel("Pixel count")
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap=plt.cm.gray)
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()



def multiotsu_filter_segmentation(array):
        
    #filters
    median = cv.medianBlur(np.uint16(array),3)
    mean = cv.blur(np.uint16(array),(3,3))
    gauss = cv.GaussianBlur(np.uint16(array),(3,3),0)
    
    #thresholds
    median_thresh = threshold_multiotsu(median)
    mean_thresh = threshold_multiotsu(mean)
    gauss_thresh = threshold_multiotsu(gauss)
    
    #segmentations
    median_seg = np.digitize(median, bins=median_thresh).astype('uint8')
    mean_seg = np.digitize(mean, bins=mean_thresh).astype('uint8')
    gauss_seg = np.digitize(gauss, bins=gauss_thresh).astype('uint8')
    
    return  median_seg,mean_seg,gauss_seg
    
def display_multiotsu_filter(array):
    
    multi_segment,_ = multiotsu_segmentation(array)
    
    median_seg,mean_seg,gauss_seg = otsu_filter(array)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(15, 8))
    axes = axes.ravel()

    axes[0].imshow(multi_segment, cmap=plt.cm.gray)
    axes[0].set_title('Multiotsu image before filtering ')

    axes[1].imshow(gauss_seg, cmap=plt.cm.gray)
    axes[1].set_title('Gaussian filter')

    axes[2].imshow(mean_seg, cmap=plt.cm.gray)
    axes[2].set_title('Mean filter')

    axes[3].imshow(median_seg, cmap=plt.cm.gray)
    axes[3].set_title('Median filter')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()