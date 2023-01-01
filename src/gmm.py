import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import scipy.stats as sps
from scipy.ndimage import convolve
import skimage
from scipy import ndimage
from scipy.spatial.distance import cdist
from nibabel.testing import data_path
import nibabel as nib
from sklearn.mixture import GaussianMixture
import pandas as pd
import skimage.measure
import pydicom as dicom
import sys
import cv2




def GMM(data_case1):
    
    #denoising
    data= cv2.medianBlur(np.uint16(data_case1),5)
    
    #save all the pixels greater than 0 as T, with the shape (n,)
    T = data[data > 0]
    
    #defining an instance of the GaussianMixture model,fitting it into the reshaped data T,this give the parameters of the models 
    # i.e the means and the covariance of the model 
    model = GaussianMixture(2,covariance_type='spherical').fit(T.reshape((T.shape[0], 1)))
    
    # means is the means vector of the model 
    means =  model.means_
    
    #variance is the covariance of the model 
    variance = model.covariances_

    #set the first threshold to be the minimum of the means,since the means is vector
    thresh_1 = np.min(means) 

    #setting the second threshold to be equal the difference between the maximum of the "means" and the squareroot of 
    #  the corresponding variance  where the location gives the maximum value from the mean.
    thresh_2 = np.max(means) - np.sqrt(variance[np.argmax(means)])
    
    #thresh_abd_1 is the same as thresh_2
    thresh_abd_1 = np.max(means) - np.sqrt(variance[np.argmax(means)])

    #setting thresh_abd_2 to be equal the sum of the maximum of the "means" and the squareroot of  
    # the corresponding variance  where the location gives the maximum value from the mean.
    thresh_abd_2 = np.max(means) + np.sqrt(variance[np.argmax(means)])
    
    #we set the the shape (row and column) of the argument (data_case1) of our function to be l and d
    l,d = data_case1.shape

    #--------- Abdomen :---------------------------------------------------------------------------------------------------------
    
    # Array of zeros  having the same shape as the argument 'data_case' 
    adbomen = np.zeros((l, d, 3))

    #setting the first channel of the adbomen  to be between the threshold (thresh_abd_1< data_case1 < thresh_abd_2 ) multiply the output by 255
    adbomen[:,:,0] = ((data_case1> thresh_abd_1)*1)*((data_case1 < thresh_abd_2)*1 )*255

    #setting the second channel of the adbomen to be between the threshold (thresh_abd_1< data_case1 < thresh_abd_2 ) multiply the output by 255
    adbomen[:,:,1] = ((data_case1> thresh_abd_1)*1)*((data_case1 < thresh_abd_2)*1 )*255

    #setting the third channel of the adbomen  to be between the threshold (thresh_abd_1< data_case1 < thresh_abd_2 ) multiply the output by 255
    adbomen[:,:,2] = ((data_case1> thresh_abd_1)*1)*((data_case1 < thresh_abd_2)*1 )*255
    
    #The function cv.threshold is used to apply the thresholding. The first argument is the source image, which is the adbomen .
    # The second argument is the threshold value which is used to classify the pixel values. 
    # The third argument is the maximum value which is assigned to pixel values exceeding the threshold
    #OpenCV provides different types of thresholding which is given by the fourth parameter of the function. 
    # thresholding as used below is done by using the type cv.THRESH_BINARY. The function transforms the adbomen image to a binary image according to 
    # the formulae it gives the 𝚖𝚊𝚡𝚅𝚊𝚕𝚞𝚎(255) if the adbomen>200 and 0 otherwise.
    ret,img = cv2.threshold(adbomen,200,255,cv2.THRESH_BINARY)
    
    #convert the datatype of the first channel of adbomen to uint8,which is gray
    gray = np.uint8(adbomen[:,:,0])

    
    #------------------------------------------------------------------------------------------------------------------
    #contors is a python list of all the contours in the image. Each individual contour is a Numpy array
    # of (x,y) coordinates of boundary points of the object.
    #hierarchy is the optional output vector which is containing the information about the image topology
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    
    
    #------------------------------------------------------------------------------------------------------------------
    #display the number of contors which are found inside the image 
    #print("Number of contours = " + str(len(contours)))
    #print out the contours of the first index,which is a numpy array of (x,y) coordinates
    #print(contours[0])
    #if we plot the  x and y coordinates we are going to get the boundary of the contours,
    # pass the contours to the method drawcontors to draw or join all the coordinates of those contours 
    #cv2.drawContours(original_image,contours,-1,(0, 255, 0),3) #-1 will let it draw all the contours finds in the image #0 is the first contour found inside the image
    #------------------------------------------------------------------------------------------------------------------
    
    
    segment= cv2.drawContours(gray, contours, 3, (0,255,0), 3)
    # Map component labels to  RGB value , 0-255 is the RGB range in OpenCV
    #in labels above we classes that ranges from 0-1,we try are trying to converts it number that ranges between 0-255,
    label_hue = np.uint8(255*255*255*segment/np.max(segment))
    
    #we are trying to compute one of the channel of the image 
    blank_ch = 255*np.ones_like(label_hue)
    
    #cv2.merge takes single channel images and combines them to make a multi-channel image.
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch]) 
    
    # Converting cvt to BGR, we  use cv2. cvtColor()  to convert an image from one color space to another.
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0

    return  gray, labeled_img
    #plt.imshow(segment, cmap=plt.cm.gray)




def morphological(data):
    #Performing the segmentation
    _,labeled_img=GMM(data)
    kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)) #cv.MORPH_RECT
    dilation = cv2.dilate(labeled_img,kernel,iterations = 1)
    erosion = cv2.erode(labeled_img,kernel,iterations=1)
    opening = cv2.morphologyEx(labeled_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(labeled_img, cv2.MORPH_CLOSE, kernel)
    
    return dilation,erosion,opening,closing

def display_morphology(data):
    dilation,erosion,opening,closing =morphological(data)
    f, axarr = plt.subplots(2,2,figsize=(10,10))
    axarr[0,0].imshow(dilation, cmap=plt.cm.gray)
    axarr[0, 0].set_title('Dilation Image')

    axarr[0,1].imshow(erosion, cmap=plt.cm.gray)
    axarr[0, 1].set_title('Erosion Image')

    axarr[1,0].imshow(opening, cmap=plt.cm.gray)
    axarr[1, 0].set_title('Opening Image')

    axarr[1,1].imshow(closing, cmap=plt.cm.gray)
    axarr[1, 1].set_title('Closing Image')
    plt.show()
