
import cv2
import numpy as np
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

#make interface for image

#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass
    
def make_interface(window_name):
    """Make an interface for an image"""
    
    file_path = easygui.fileopenbox()

    cv2.namedWindow(window_name)
    cv2.createTrackbar('PPMMultiplier',window_name,10,20, nothing)
    cv2.createTrackbar('STD_bounds_Lower',window_name,25,100,nothing)
    cv2.createTrackbar('STD_bounds_Upper',window_name,75,100,nothing)
    cv2.createTrackbar('ImFill',window_name,0,1,nothing)
    cv2.createTrackbar('HighPassFrequency',window_name,0,100,nothing)
    cv2.createTrackbar('Contrast',window_name,0,100,nothing)
    cv2.createTrackbar('Threshold',window_name,0,100,nothing)

    return file_path

def get_values(window_name):
    """get the values from the interface for a particular window"""
    PPMMultiplier = cv2.getTrackbarPos('PPMMultiplier',window_name)
    STD_bounds_Lower = cv2.getTrackbarPos('STD_bounds_Lower',window_name)
    STD_bounds_Upper = cv2.getTrackbarPos('STD_bounds_Upper',window_name)
    ImFill = cv2.getTrackbarPos('ImFill',window_name)
    HighPassFrequency = cv2.getTrackbarPos('HighPassFrequency',window_name)
    Contrast = cv2.getTrackbarPos('Contrast',window_name)
    Threshold = cv2.getTrackbarPos('Threshold',window_name)
    return PPMMultiplier, STD_bounds_Lower, STD_bounds_Upper, ImFill, HighPassFrequency, Contrast, Threshold

    
#run the interface

def run_pipeline(source_data, PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Contrast, Threshold):

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
        print(STD_bounds_lower,STD_bounds_upper)
        ZNormals = (Zs - mean_z)/ZSTD
        ZNormals = [z if (z > STD_bounds_lower) else (STD_bounds_lower) for z in ZNormals] #lazy. Force to -2 or 2 range
        ZNormals = [z if (z < STD_bounds_upper) else STD_bounds_upper for z in ZNormals ] #this has 95% of data
        ZNormals = [(z/(STD_bounds_upper-STD_bounds_lower) + 0.5) * 255 for z in ZNormals] #/5 + 0.5 for -2 to 2. Check if can be further normalized

        for k, depth in enumerate(ZNormals):
            XIndex = XsPixels[k]
            YIndex = YsPixels[k]
                
            image[XIndex,YIndex] = depth #actual data is colored
    
    return(image)
    

def run_interface(window_name="slider control for input data"):
    """Run the interface"""
    file_path = make_interface(window_name)
    source_data = read_fingerprints(file_path)

    
    while(1):
        PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Contrast, Threshold = get_values(window_name)
        PPMMultiplier = PPMMultiplier/10
        STD_bounds_lower = (STD_bounds_lower/10) - 5
        STD_bounds_upper = (STD_bounds_upper/10) - 5

        image = run_pipeline(source_data, PPMMultiplier, STD_bounds_lower, STD_bounds_upper, ImFill, HighPassFrequency, Contrast, Threshold)
        cv2.imshow('image',image.astype(np.uint8))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
    cv2.destroyAllWindows()

run_interface()
#change the image
