from scipy import ndimage as ndi
from skimage import feature
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




def watershed_kmeans_segmentation(array):  
    
    #Denoising the array
    #array = filters.median(array, footprint=np.ones((5,5))).astype('float32')
    array =  ndi.gaussian_filter(array,1,mode='nearest').astype('uint8')
    
    #Finding the edges of the array using Canny Edge detector
    edges = feature.canny(array, sigma=1)
    
    #Converting the edges into a landscape using distance transform
    dt = distance_transform_edt(~edges)
    
    #find the locations of the fountains
    local_max = feature.peak_local_max(dt, indices=False,min_distance=5)
    
    #visualize the peak index 
    #peak_idx = feature.peak_local_max(dt, indices=True,min_distance=5)
    #plt.plot(peak_idx[:,1],peak_idx[:,0],'r')
    #plt.imshow(dt)
    
    #label each of these features 
    markers = measure.label(local_max,background=0)
    
    
    #watershed
    labels = watershed(-dt, markers)
    
    #first visualization
    #plt.imshow(segmentation.mark_boundaries(array,labels))
    
    #second visualization
    #segment = color.label2rgb(labels, image=array, kind='avg')
    #plt.imshow(segment, cmap='gray')

    # regionprops returns a list of properties for each labeled region
    regions = measure.regionprops(labels, intensity_image=array)
    region_means = [r.mean_intensity for r in regions]
    region_means = np.array(region_means).reshape(-1,1)
    #plt.hist(region_means, bins=20)
    
    #we want this process to be automated that is why we are using scikit-learnn K-means in order to do a clustering 
    #of the background and the foreground intensities
    model = KMeans(n_clusters=2,algorithm='elkan')
    region_means = np.array(region_means).reshape(-1,1)
    model.fit(np.array(region_means))
    
    
    #print(model.cluster_centers_) #this is going to give the centers for the two classes , these are clusters for 
    #our foreground and background
    
    #now we ask the model to predict label for each of my regions, and it will give the label foreground or background
    bg_fg_labels = model.predict(region_means)
    
    #Now we want to label the image appropriately
    classified_labels = labels.copy()
    #we take the combination of our predicted labels with different regions, we relabel the image according to the 
    #coordinate of each region and assign that to whether is background or foreground
    for bg_fg ,region in zip(bg_fg_labels,regions):
        
        #relabelling the coordinate according to the coordinate of each region and assign that to whether 
        #is background or foreground
        classified_labels[tuple(region.coords.T)] = bg_fg
    labels = np.zeros(classified_labels.shape) 
    
    if (classified_labels[1500:1510,2000:2010]==1).all():
            labels[classified_labels==0]=1
            labels[classified_labels==1]=0
            return labels
    else:
        return classified_labels
        
def display_watershed(array):
    classified_labels=watershed_kmeans_segmentation(array)

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
    ax1.imshow(array, cmap=plt.cm.gray)
    ax1.set_title('Original Image')

    ax2.imshow(classified_labels, cmap=plt.cm.gray)
    ax2.set_title('Watershed segmentation')
    plt.show()