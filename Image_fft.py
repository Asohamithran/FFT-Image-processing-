import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig
from pathlib import Path
import scipy
from PIL import Image,ImageOps
image_path = Path("Sambhabana Mohanty.jpg")
## get the image and convert into a matrix
image_1= Image.open(image_path)
matrix1=np.asarray(image_1)
## convert the rgb image to grayscale image
Image_2= ImageOps.grayscale(image_1)
#Image_2.show()
matrix_2= np.asarray(Image_2)
fft_matrix_2= np.fft.fft2(matrix_2)
Ifft_matrix_2= np.fft.ifft2(fft_matrix_2)
fft_matrix_2= np.log(abs(np.fft.fftshift(fft_matrix_2)))
print(np.max(matrix_2))
fft_image= Image.fromarray(fft_matrix_2)import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig
from pathlib import Path
import scipy
from PIL import Image,ImageOps


image_path = Path("Sambhabana Mohanty.jpg")
## get the image and convert into a matrix
image_1= Image.open(image_path)
matrix1=np.asarray(image_1)
## convert the rgb image to grayscale image

Image_2= ImageOps.grayscale(image_1)
#Image_2.show()
# convert the image to a matrix and perform FFT
matrix_2= np.asarray(Image_2)
fft_matrix_2= np.fft.fft2(matrix_2)
Ifft_matrix_2= np.fft.ifft2(fft_matrix_2)
fft_matrix_2= 20*np.log(abs(np.fft.fftshift(fft_matrix_2))) # upscaled_FFT
#print(np.max(matrix_2))
fft_image= Image.fromarray(fft_matrix_2)
fft_image.convert("RGB").save("Upscaled_fft.PNG") #convert the image to b/w to RGB channel
Ifft_matrix_2=abs(Ifft_matrix_2)
iFFT_image= Image.fromarray(Ifft_matrix_2)
# iFFT_image.show()
print("\nshowing the image in freq space and real space")
plt.figure()
plt.subplot(2,2,1)
plt.imshow(matrix1)
plt.subplot(2,2,2)
plt.imshow(matrix_2,cmap="gray")
plt.subplot(2,2,3)
plt.imshow(fft_matrix_2)
plt.subplot(2,2,4)
plt.imshow(Ifft_matrix_2,cmap="gray")
plt.show()

#fft_image.show()
iFFT_image= Image.fromarray(abs(Ifft_matrix_2))
iFFT_image.show()
