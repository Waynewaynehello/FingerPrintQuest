#Python 3 

"""Import a text file 'fingerprints.txt' and convert the list of comma seperated, X, Y, and Z coordinate values into a 3D Numpy array"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import statistics
import fingerprint_enhancer #uses gabor filters


def read_fingerprints(filename):
    """Read a file of fingerprints into a 3D Numpy array"""
    with open(filename) as f:
        fingerprints = f.readlines()
    fingerprints = [x.strip() for x in fingerprints]
    fingerprints = [x.split() for x in fingerprints]
    fingerprints = np.array(fingerprints, dtype=float)
    return fingerprints

from skimage.morphology import reconstruction

def imfill(img):
    # https://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale
    # Use the matlab reference Soille, P., Morphological Image Analysis: Principles and Applications, Springer-Verlag, 1999, pp. 208-209.
    #  6.3.7  Fillhole
    # The holes of a binary image correspond to the set of its regional minima which
    # are  not  connected  to  the image  border.  This  definition  holds  for  grey scale
    # images.  Hence,  filling  the holes of a  grey scale image comes down  to remove
    # all  minima  which  are  not  connected  to  the  image  border, or,  equivalently,
    # impose  the  set  of minima  which  are  connected  to  the  image  border.  The
    # marker image 1m  used  in  the morphological reconstruction by erosion is set
    # to the maximum image value except along its border where the values of the
    # original image are kept:

    seed = np.ones_like(img)*255
    img[ : ,0] = 0
    img[ : ,-1] = 0
    img[ 0 ,:] = 0
    img[ -1 ,:] = 0
    seed[ : ,0] = 0
    seed[ : ,-1] = 0
    seed[ 0 ,:] = 0
    seed[ -1 ,:] = 0

    fill_img = reconstruction(seed, img, method='erosion')

    return fill_img

"""Write a CV2 function to take a list of images and tile them into one big image. Add a text label to each image""" 

def tileImages(images, labels, fontSize=10, fontColor=(255,255,255), bgColor=(0), font=cv2.FONT_HERSHEY_SIMPLEX, bottomLeftCornerOfText=(0,0), fontScale=0.1, lineType=2):
    """
    images: list of images
    labels: list of labels
    fontSize: font size of labels
    fontColor: font color of labels
    bgColor: background color of labels
    font: font of labels
    bottomLeftCornerOfText: bottom left corner of labels
    fontScale: font scale of labels
    lineType: line type of labels
    """
    # Get the width and height of the images
    width = images[0].shape[1]
    height = images[0].shape[0]
    
    # Get the number of images
    numImages = len(images)
    
    # Get the number of rows and columns needed
    rows = math.ceil(math.sqrt(numImages))
    cols = math.ceil(numImages/rows)
    
    # Create a blank image
    blankImage = np.zeros((height*rows, width*cols), np.uint8)
    blankImage[:,:] = bgColor
    
    # Create a list of labels
    labelList = []
    for i in range(len(labels)):
        labelList.append(labels[i] + " " + str(i))
    
    # Tile the images
    for i in range(len(images)):
        # Get the row and column number
        row = math.floor(i/cols)
        col = i%cols
        
        # Place the image
        blankImage[row*height:(row+1)*height, col*width:(col+1)*width] = images[i]
        
        # Add the label
        #cv2.putText(blankImage, labelList[i], (50,50), font, 0.5, fontColor, lineType)
    
    return blankImage


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

def increase_contrast(gray, blockSize=81, k=0.1):
    
    T = cv2.ximgproc.niBlackThreshold(gray, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=81, k=0.1, binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
    grayb = (gray > T).astype("uint8") * 255
    return grayb


path = "\\" #SAVE path


#fingerprints = read_fingerprints('fingerprintsdata1.txt')
#fingerprints = read_fingerprints('fingerprintsdata2.txt')
#fingerprints = read_fingerprints('coin.txt')
fingerprints = read_fingerprints('fingerprintsdata3.txt')
#get individual matrixes for easyness

Xs = fingerprints[:,0] #SWAPPED FOR THIS MACHINE
Ys = fingerprints[:,1]  #from CMM coordinates
Zs = fingerprints[:,2] 

#get min max X,Y,Z. 
XMin,XMax = Xs.min(), Xs.max()
YMin,YMax = Ys.min(), Ys.max()
ZMin,ZMax = Zs.min(), Zs.max()

PixelsPerMM = (len(Xs) / ((XMax-XMin) * (YMax-YMin)))**(1/2)

PixelsPerMM = PixelsPerMM * 0.25 #lower values reduce noise, hide data since pixels not evenly dispersed
print('PPMM = {}, total points = {}'.format(PixelsPerMM,len(Xs)))

XPixels = int((XMax-XMin) * PixelsPerMM) + 1 #a NUMBER
YPixels = int((YMax-YMin)*PixelsPerMM) + 1

print('len of Xpixels is {}, Ypixels is {}'.format(XPixels,YPixels))

#convert X location into closest pixel value
#leads to noise I think because points not evenly spaced
XsPixels = [int(round(x)) for x in ((Xs - XMin) *PixelsPerMM)] #an ARRAY
YsPixels = [int(round(y)) for y in ((Ys - YMin) *PixelsPerMM)]

#generate range of possible normalized images
#attempted normalization from Stephen Joy, ResearchGate

mean_z = Zs.mean()
ZSTD = statistics.stdev(Zs)
normalized_images = [] #up to this point, image is identical in pipeline
normalized_names = []

for i in range(-6,2,1): #bounds_lower/2 for normalizing and removing outliers
    for j in range(1,8,1): #bounds_upper. Closer upper to lower, more data thrown out
        bounds_lower, bounds_upper = i/2, j/2
        if bounds_lower < bounds_upper:
            image = np.ones ((XPixels+1, YPixels+1)) * 0 #multiply by zero to make black, 255 white
            ZNormals = (Zs - mean_z)/ZSTD #ZScore; mean will be zero when range from -2 to 2
            #ZNormals = [z if z > (bounds_lower * -1) else (bounds_lower * -1) for z in ZNormals] #lazy. Force to -2 or 2 range
            ZNormals = [z if z > (bounds_lower) else (bounds_lower) for z in ZNormals] #lazy. Force to -2 or 2 range
            ZNormals = [z if z < bounds_upper else bounds_upper for z in ZNormals ] #this has 95% of data
            ZNormals = [(z/(bounds_upper-bounds_lower) + 0.5) * 255 for z in ZNormals] #/5 + 0.5 for -2 to 2. Check if can be further normalized
            for k, depth in enumerate(ZNormals):
                XIndex = XsPixels[k]
                YIndex = YsPixels[k]
                
                image[XIndex,YIndex] = depth #actual data is colored

            
            #image = imfill(image) #WARNING WARNING WARNING MAYBE THIS IS BAD    
            

            name = "Normalized with lower cutoff of {} and upper of {} ".format(bounds_lower,bounds_upper)
            normalized_images.append(image)
            normalized_names.append(name)
            print('currently trying out: {}'.format(name))
            cv2.imwrite(path+name+'.bmp',image)

            #1 and 1 is a good one maybe. Also maybe -4 and 5 just to show all data

tiled = tileImages(normalized_images,normalized_names)
cv2.imshow('tiled',tiled)

#for loop for Canny parameters. Do the same for normalization and highpass

edge_combos = [] #save list of edge combo images for faster testing
edge_names = []

filled = imfill(image) #attempt to remove white or black spots
highpassed = highpass(filled, 1)


for aperture in range(5,8,2):
    for lower in range(25,250,50):
        for upper in range(100,350,50):
            if lower < upper:
                name = "canny lower is {} canny upper is {} canny aperture is{}".format(lower,upper,aperture)
                edge = cv2.Canny(highpassed.astype(np.uint8), lower, upper, 
                    apertureSize=aperture)
                edge_combos.append(edge)
                    
                edge_names.append(name)

                cv2.imwrite(path+name+'.jpg',edge)
                    #print(name," failed")




# Applying the Canny Edge filter
# with Custom Aperture Size
edge = edge = cv2.Canny(highpassed.astype(np.uint8), t_lower, t_upper, 
                 apertureSize=aperture_size)



#enhanced_filled = fingerprint_enhancer.enhance_Fingerprint(filled.astype(np.uint8))

#cv2.imshow('depthmap', np.uint8(image)) #EMPTY SPACE CAUSING BLACK
#cv2.imshow('filled depth map', np.uint8(filled)) #EMPTY SPACE CAUSING BLACK
#cv2.imshow('highpassed', np.uint8(highpassed))
#cv2.imshow('Edge detected from highpass', np.uint8(edge))


try: 
    enhanced = fingerprint_enhancer.enhance_Fingerprint(highpassed.astype(np.uint8))

    cv2.imshow('enhanced from highpass', np.uint8(enhanced))
except Exception:

    print('that random enhance fingerprint module failed')
#cv2.imshow('enhanced from original filled image', np.uint8(enhanced_filled))
    
                                                           



print('ding ding ding')
    
