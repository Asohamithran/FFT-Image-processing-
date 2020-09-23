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
fft_image= Image.fromarray(fft_matrix_2)
#fft_image.show()
iFFT_image= Image.fromarray(abs(Ifft_matrix_2))
iFFT_image.show()
