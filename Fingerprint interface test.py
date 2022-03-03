
import cv2
import numpy as np
import math
import easygui
import statistics
from skimage.morphology import reconstruction



#read a file

def read_fingerprints(filename):
    """Read a file of fingerprints into a 3D Numpy array"""
    with open(filename) as f:
        fingerprints = f.readlines()
    fingerprints = [x.strip() for x in fingerprints]
    fingerprints = [x.split() for x in fingerprints]
    fingerprints = np.array(fingerprints, dtype=float)
    return fingerprints

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass
    
def make_interface(window_name):
    """Make an interface for controlling aspects of an image"""
    
    file_path = easygui.fileopenbox()

    cv2.namedWindow(window_name) #default, max. All have min of 0
        
    cv2.createTrackbar('SizeCanv.',window_name,10,30, nothing)
    cv2.createTrackbar('MinSigma',window_name,25,100,nothing)
    cv2.createTrackbar('MaxSigma',window_name,75,100,nothing)
    cv2.createTrackbar('fill gaps',window_name,1,5,nothing)
    cv2.createTrackbar('HighPass',window_name,10,100,nothing)
    cv2.createTrackbar('Contrast',window_name,0,100,nothing)
    cv2.createTrackbar('Threshold',window_name,0,255,nothing)

    return file_path

def imfill2(image,radius,blank=0):
    
    "https://datascience.stackexchange.com/questions/51375/how-to-replace-nan-values-for-image-data" 
    mask = (np.asarray((image==blank), dtype="uint8"))
    dst = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)

    return dst 

def threshold(image,Threshold,set_to=255,method=cv2.THRESH_BINARY):
    ret, thresh_img = cv2.threshold(image, Threshold, set_to, method)
    return thresh_img

def get_values(window_name):
    """get the values from the interface for a particular window"""
    PPMMultiplier = cv2.getTrackbarPos('SizeCanv.',window_name)
    STD_bounds_Lower = cv2.getTrackbarPos('MinSigma',window_name)
    STD_bounds_Upper = cv2.getTrackbarPos('MaxSigma',window_name)
    ImFill = cv2.getTrackbarPos('fill gaps',window_name)
    HighPassFrequency = cv2.getTrackbarPos('HighPass',window_name)
    Contrast = cv2.getTrackbarPos('Contrast',window_name)
    Threshold = cv2.getTrackbarPos('Threshold',window_name)
    return PPMMultiplier, STD_bounds_Lower, STD_bounds_Upper, ImFill, HighPassFrequency, Contrast, Threshold

    
#run the actual image processing dependent on interface

def run_pipeline(source_data, PPMMultiplier, STD_bounds_lower,
                 STD_bounds_upper, ImFill, HighPassFrequency,
                 Contrast, Threshold):

    Xs = source_data[:,0] #SWAPPED FOR THIS MACHINE
    Ys = source_data[:,1]  #from CMM coordinates
    Zs = source_data[:,2] 

    XMin,XMax = Xs.min(), Xs.max()
    YMin,YMax = Ys.min(), Ys.max()
    ZMin,ZMax = Zs.min(), Zs.max()

    PixelsPerMM = (len(Xs) / ((XMax-XMin) * (YMax-YMin)))**(1/2)
    PixelsPerMM = PixelsPerMM * PPMMultiplier #lower values reduce noise, hide data since pixels not evenly dispersed
    #print('PPMM = {}, total points = {}'.format(PixelsPerMM,len(Xs)))

    XPixels = int((XMax-XMin) * PixelsPerMM) + 1 #a NUMBER
    YPixels = int((YMax-YMin)*PixelsPerMM) + 1

    #print('len of Xpixels is {}, Ypixels is {}'.format(XPixels,YPixels))


    #convert X location into closest pixel value
    #leads to noise I think because points not evenly spaced

    XsPixels = [int(round(x)) for x in ((Xs - XMin) *PixelsPerMM)] #an ARRAY
    YsPixels = [int(round(y)) for y in ((Ys - YMin) *PixelsPerMM)]

    #attempted normalization from Stephen Joy, ResearchGate

    mean_z = Zs.mean()
    ZSTD = statistics.stdev(Zs)
    
    image = np.ones ((XPixels+1, YPixels+1)) * 0 #multiply by zero to make black, 255 white
    
    if STD_bounds_lower < STD_bounds_upper:
        #print(STD_bounds_lower,STD_bounds_upper)
        ZNormals = (Zs - mean_z)/ZSTD
        ZNormals = [z if (z > STD_bounds_lower) else (STD_bounds_lower) for z in ZNormals] #lazy. Force to -2 or 2 range
        ZNormals = [z if (z < STD_bounds_upper) else STD_bounds_upper for z in ZNormals ] #this has 95% of data
        ZNormals = [(z/(STD_bounds_upper-STD_bounds_lower) + 0.5) * 255 for z in ZNormals] #/5 + 0.5 for -2 to 2. Check if can be further normalized

        for k, depth in enumerate(ZNormals):
            XIndex = XsPixels[k]
            YIndex = YsPixels[k]
                
            image[XIndex,YIndex] = depth #actual data is colored

    image = image.astype(np.uint8)

    if ImFill: #acts as both on/off and radius of search 
        image = imfill2(image,ImFill)
    
    if HighPassFrequency: #if 0 then not used
        image = highpass(image, HighPassFrequency)

    if Threshold:
        image = threshold(image,Threshold)
        
    return(image)
    

def run_interface(window_name="Fingerprint madness"):
    """Run the interface"""
    file_path = make_interface(window_name)
    source_data = read_fingerprints(file_path)

    
    while(1):
        PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Contrast, Threshold = get_values(window_name)
        PPMMultiplier = PPMMultiplier/10
        STD_bounds_lower = (STD_bounds_lower/10) - 5 #default: -2.5 to 2.5, total range of -5 to 5
        STD_bounds_upper = (STD_bounds_upper/10) - 5
        HighPassFrequency = HighPassFrequency/10
        ImFill = max((ImFill*10)-9,0) #default: 1, range 0.1 to 10, if 0 then not used
        
        if Threshold: #!=0
            newThreshold = abs(Threshold - 256) #0 still 0, but 1 is now white #lazy
            Threshold=newThreshold #for some reason, won't work without this             
        
        image = run_pipeline(source_data, PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Contrast, Threshold)
        
        cv2.imshow('image',image.astype(np.uint8))
        #cv2.imshow(window_name,image.astype(np.uint8))
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return(image)
            #break
        
    cv2.destroyAllWindows()

image = run_interface() #return image at break for use in testing
#change the image
