# -*- coding: utf-8 -*-
"""
Particle image analysis using skimage package

program based on
https://github.com/smahala02/Materials-Science-Image-Analysis
https://github.com/smahala02/Materials-Science-Image-Analysis/blob/main/Image%20Analysis%20for%20Materials%20Science.ipynb

Image information:
Ni superalloys are amazing materials. Whereas most alloys become softer as the 
temperature increases, the alloys maintain their strength at very high 
temperatures and sometimes even get stronger! This ability, combined with 
their excellent corrosion resistance and toughness have made them the alloy of 
choice in high temperature applications like jet engines.

The image in the file Coarse_250kx.tif is a scanning electron microscope (SEM) 
image of RR1000, a nickel superalloy developed by Rolls-Royce. In this alloy, 
the high temperature strength of the alloys originates from the presence of 
small particles of an ordered intermetalic phase called (gamma prime).

The image in file Coarse_250kx.tif is of a RR1000 sample where the particles 
have been etched away to leave behind dark holes which have the same shape and 
size as the particles.


General steps:
    
- load image
- use contrasting and thresholding to distinguish particle from background to
  make a binary image (particles vs background).  This step may be challenging,
  depending on the image.  Decisions may need to be made about what to include
  and how to deal with depth of field effects.
- identify connected pixels for each shape
- use the information to compute sizes of the particles
- use an image of an object with a known size to scale the pixel sizes
- compute some statistsics about the particle sizes and distribution

The skimage package has routines to assist with some of these steps.

*****
This program started by converting Python code written using a Jupyter
notebook for an assignment - see links above.
Note that this program has not been refined (extra code, plots missing some
labels and titles, commented out code, etc.).  It also contains some extra code, 
either left in or commented out, that is used for testing or to demonstrate 
some features.
*****

Created on Mon Sep 22, 2025

Last modified:  Oct. 6 2025

@author: FairMC
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, measure
from skimage.filters import threshold_otsu, try_all_threshold,threshold_mean

def main():

    # read in image file (this file was in grayscale so no conversion is required)
    image1 = io.imread('frame_0410_8bit.TIF')
    image = 255 - image1 # this inverts the image if using a black background and making the droplets white
    # plot image file
    plt.figure()
    plt.imshow(image,cmap='gray')
    
    
    # Next try to determine thresholding needed to convert image
    # to a binary image of just the objects (Here objects are darker)
    # Various thresholding methods exist (global and local) and
    # various contrasting methods exits.
    # Note for 8-bit grayscale (0 = black to 255 = white)
    
    # A histogram is sometimes helpful to determine a threshold, 
    # so create a histogram of the intensities on the image
    hist = exposure.histogram(image)
    plt.figure()
    plt.fill_between(hist[1], hist[0], alpha=0.5, color='b')
    
    # Next two lines give an alternative way to make a histogram using numpy
    # hist, bin_edges = np.histogram(image, bins=128)  # Example with 128 bins
    # plt.fill_between(bin_edges[:-1], hist, alpha=0.5, color='b')
    
    # Set the number of bins and adjust x-axis limits
    plt.xlim([0, 256])  # x-axis limit to cover the intensity range
    plt.ylim([0, hist[0].max() * 1.1])  # Adding some space above the max value 
                                        # for better visibility
    plt.xlabel('Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Intensity Histogram')
    plt.show()  # Display the histogram
    
    
    # examine various global thresholding methods available in skimage
    fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
    plt.show()
    
    
    # # brief look at a local thresholding method
    # import skimage as ski
    # block_size = 35
    # local_thresh = ski.filters.threshold_local(image, block_size, offset=10)
    # binary_local = image > local_thresh
    # plt.figure()
    # plt.imshow(binary_local,cmap='gray')
    # plt.show()
    
    
    # Try manual thresholding based on histogram
    # Select 90 as the threshold
    # If value is larger than 90 then True (white) else False (black)
    man_thresh = 90
    pixel = image > man_thresh # result is a bolean array of True or False 
    print('manual threshold - set using histogram = ',man_thresh)
    
    # create plots
    f,images_1 = plt.subplots(1,2)
    images_1[0].imshow(pixel, cmap='gray')
    images_1[1].imshow(image, cmap='gray')
    images_1[0].set_title('Manual Image')
    images_1[1].set_title('Orginal Image')
    plt.show()
    
    
    # use a builtin thresholding method in skimage
    ots = threshold_otsu(image)
    print('threshold value from otsu in skimage =',ots)
    
    pixel_1 = image > ots
    
    # create plots
    f,images_2 = plt.subplots(1,2)
    images_2[0].imshow(pixel_1,cmap='gray')
    images_2[1].imshow(image,cmap='gray')
    images_2[0].set_title('Otsu Image')
    images_2[1].set_title('Orginal Image')
    plt.show()
    
    # Lines below use the builtin mean method for the threshold
    # from skimage.filters import threshold_mean
    # thmean = threshold_mean(image)
    # print('threshold value from otsu in skimage ',thmean)
    # pixel_m = image > thmean
    # # create plots
    # f,images_2 = plt.subplots(1,2)
    # images_2[0].imshow(pixel_m,cmap='gray')
    # images_2[1].imshow(image,cmap='gray')
    # images_2[0].set_title('Mean Image')
    # images_2[1].set_title('Orginal Image')
    # plt.show()
    
    
    # # create plots to compare manual and otsu results
    # f,images_3 = plt.subplots(1,2)
    # images_3[0].imshow(pixel_1,cmap='gray')
    # images_3[1].imshow(pixel,cmap='gray')
    # images_3[0].set_title('Otsu Image')
    # images_3[1].set_title('Manual Image')
    # plt.show()
    
    
    # evaluate the difference between the manual and otsu thresholding methods
    # first convert boolean arrays to integer arrays
    a = pixel.astype(int)
    b = pixel_1.astype(int)
    Diff = a - b
    # create plot
    plt.figure()
    plt.imshow(Diff,cmap='gray')
    
    
    # Choose somewhat arbitrarily to use ots threshold (otsu is a common method)
    # Now use skimage measure.label function to label connected regions 
    #  of an integer array
    # mean_ = threshold_mean(image)
    # pixel_2 = image < mean_  # this is the same as pixel_1 above
    
    # find and label pixels of connected or touching dark pixels (particles)
    dark = measure.label(pixel_1) # using label function in skimage.measure
    
    # labels = np.unique(dark)  # find unique labels
    # print('number of labeled regions ',labels.size)
    # print(labels)
    
    #  skimage can use the labeled file "dark" to get properties of the
    # connected regions
    
    # next several lines get an print the properties of the 'partcles'
    # Note that pandas is uded to print the results in a nice table
    prop_val = measure.regionprops_table(dark,
                              properties=['label', 'num_pixels','centroid',
                                          'area','equivalent_diameter'])
    import pandas as pd  
    data = pd.DataFrame(prop_val)
    print()  
    print('****************************************')
    print('Results based on number of pixels only - not scaled')
    print(data.to_string())
    print('****************************************')
    
    
    
    # use a known scale to scale the pixel sizes
    # Load the scale bar image
    scale = io.imread('scalebar.png')
    
    # plot image of scale bar uncorrected
    # plt.figure()
    # plt.imshow(scale,cmap='gray')
    # plt.show()
     
    # Convert to binary to isolate the white scale bar
    scale_white = scale > 100  #254  # Assuming the scale bar is white
    plt.figure()
    plt.imshow(scale_white, cmap='gray')
    plt.show()
    
    # Get the dimensions of the scale bar
    # The scale bar is between pixel 20 and 460 - using the inage of the bar
    PixelLength = 460 - 20  # Length of the scale bar in pixels
    print()
    print('Using the image of the scale bar')
    print('One micrometer is: ' + str(PixelLength) + ' pixels')
    print()
    
    # Scale bar is 1 micron in length
    # Conversion factor (1 micrometer = 1000 nanometers)
    # Convert the equivalent diameter from pixels to nanometers
    nm_per_pixel = 1000 / PixelLength  # How many nanometers one pixel represents
    
    
    
    # Assuming 'dark' has the labeled 'particles', get the properties of
    # all labeled regions
    props = measure.regionprops(dark)  # Get properties of all labeled regions
    dia = []  # To store the equivalent diameters of particles 
    number = []    # To store the particle numbers
    
    # Loop through all regions
    for n, region in enumerate(props):
        area = region.area  # Get the area of the region
        # Get the equivalent diameter of the region
        equivalent_diameter = region.equivalent_diameter * nm_per_pixel
        
        # Check if the area is greater than min_pixels pixels
        min_pixels = 200  #10
        if area > min_pixels:
            dia.append(equivalent_diameter)  # Append the equivalent diameter
            number.append(n)  # Append the particle number
    
    # Convert lists to numpy arrays for saving
    number = np.array(number,dtype=np.int64)
    dia = np.array(dia)
    
    print('Results for regions lager than',min_pixels,'pixels')
    print('average size (nm)',dia.mean())
    print('median size (nm)',np.median(dia))
    print('standard deviation (nm)',dia.std())
    print('number of particles ',dia.size)
    print()
    
    
    # Open file and write the data
    with open("diameter.txt", "w") as file:
        header = '\n'.join([
            "Particle  --  Equivalent diameter (nm)", 
            "-------------------------------"])  # Define the heading
        np.savetxt(file, np.transpose([number, dia]), 
                   delimiter='-----', fmt='%f', comments='# ', header=header)
    
    # Open and read the file to verify the contents
    with open('diameter.txt', 'r') as file:
        content = file.read()
    print(content)
    
if __name__ == "__main__":
    main()

