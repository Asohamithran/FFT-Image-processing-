# a simple application of gaussian blur on the given image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def lowpass(image_array):
    height=image_array.shape[0] 
    width=image_array.shape[1]
    print(height,width)
    blur_array=np.zeros((height,width))
    for i in range(height):
        for  j in range(width):
        # get each pixel in the image and then find its average
            if (i < 1 or j<1 or i+1 ==height or j+1 ==width):
                continue
            else:
                avg = image_array[i-1,j-1] +\
                      image_array[i-1,j] + \
                      image_array[i-1,j+1] + \
                      image_array[i,j+1] + \
                      image_array[i,j] + \
                      image_array[i,j+1]+ \
                      image_array[i+1,j-1] +   \
                      image_array[i+1,j]+  \
                      image_array[i+1,j+1] 
                
                blur_array[i,j]= avg/9
    return blur_array

# pass the image and then visualise the modified image 
image_file ='Sambhabana Mohanty.jpg'
img = mpimg.imread(image_file)     
gray = rgb2gray(img)    
#plt.imshow(img)
#print(gray)
plt.imshow(gray , cmap ='gray')
blurred_image=lowpass(gray)
plt.imshow(blurred_image,cmap ='gray')
plt.show()

