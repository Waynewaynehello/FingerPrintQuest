
import cv2
import numpy as np
import math
import easygui
import statistics


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

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) #default, max. All have min of 0
    #cv2.resize(window_name,(200,400))
    cv2.createTrackbar('SizeCanv.',window_name,10,30, nothing)
    cv2.createTrackbar('MinSigma',window_name,25,100,nothing)
    cv2.createTrackbar('MaxSigma',window_name,75,100,nothing)
    cv2.createTrackbar('fill gaps',window_name,1,5,nothing)
    cv2.createTrackbar('HighPass',window_name,0,200,nothing)
    cv2.createTrackbar('Mean',window_name,0,1000,nothing)
    cv2.createTrackbar('Threshold',window_name,0,255,nothing)
    cv2.createTrackbar('SwapXY',window_name,0,1,nothing)
    
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
    Mean = cv2.getTrackbarPos('Mean',window_name)
    Threshold = cv2.getTrackbarPos('Threshold',window_name)
    SwapXY = cv2.getTrackbarPos('SwapXY',window_name)
    
    return PPMMultiplier, STD_bounds_Lower, STD_bounds_Upper, ImFill, HighPassFrequency, Mean, Threshold,SwapXY

    
#run the actual image processing dependent on interface

def run_pipeline(source_dict, PPMMultiplier, STD_bounds_lower,
                 STD_bounds_upper, ImFill, HighPassFrequency,
                 Mean, Threshold,SwapXY):


    Xs = source_dict['Xs']
    Ys = source_dict['Ys']

        
    XMin = source_dict['XMin']
    YMin = source_dict['YMin']
    XMax = source_dict['XMax']
    YMax = source_dict['YMax']
        
    Zs = source_dict['Zs']
    ZMin = source_dict['ZMin']
    ZMax = source_dict['ZMax']
    
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

    if not Mean: #Controllable mean slider centered around std. Not actually the mean tho #yujie
        mean_z = Zs.mean()
    else:
        mean_z = (ZMax * Mean) + ZMin

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

    if SwapXY:
        image = np.transpose(image)
        
    return(image)
    

def run_interface(window_name="Fingerprintsss"):
    """Run the interface"""
    file_path = make_interface(window_name)
    source_data = read_fingerprints(file_path)

    
    
    Xs = source_data[:,0] 
    Ys = source_data[:,1] 
    Zs = source_data[:,2]

    XMin,XMax = Xs.min(), Xs.max()
    YMin,YMax = Ys.min(), Ys.max()
    ZMin,ZMax = Zs.min(), Zs.max()

    source_dict = {'Xs':Xs,'Ys':Ys,'Zs':Zs,'XMin':XMin,'XMax':XMax,'YMin':YMin,'YMax':YMax,'ZMin':ZMin,'ZMax':ZMax}
    
    
    print('original ZMIN {} ZMean{} ZMax {}'.format(ZMin,Zs.mean(),ZMax))
    
    
    while(1):
        PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Mean, Threshold,SwapXY = get_values(window_name)
        PPMMultiplier = PPMMultiplier/10
        STD_bounds_lower = (STD_bounds_lower/10) - 5 #default: -2.5 to 2.5, total range of -5 to 5
        STD_bounds_upper = (STD_bounds_upper/10) - 5
        HighPassFrequency = HighPassFrequency
        Mean = Mean/1000 #1000: mean is equal to max value, 1: min value. 0: true mean
        ImFill = max((ImFill*10)-9,0) #default: 1, range 0.1 to 10, if 0 then not used
        
        if Threshold: #!=0
            newThreshold = abs(Threshold - 256) #0 still 0, but 1 is now white #lazy
            Threshold=newThreshold #for some reason, won't work without this             
        
        image = run_pipeline(source_dict, PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Mean, Threshold,SwapXY)
        
        cv2.imshow('image',image.astype(np.uint8))
        #cv2.imshow(window_name,image.astype(np.uint8))
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return(image)
            #break
        
    cv2.destroyAllWindows()

image = run_interface() #return image at break for use in testing
#change the image
