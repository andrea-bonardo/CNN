import numpy as np
from PIL import Image as im
from keras.datasets import cifar10
import time

# Define a simple 1x1x3 kernel for RGB
kernel = np.array([[0.8, 0.9, 0.8], 
                   [0.9, 1, 0.9], 
                   [0.8, 0.9, 0.8]])

[training_set_inputs, training_set_outputs],[x_test, y_test] = cifar10.load_data()


image = np.asarray(x_test[0], dtype=np.uint8)
input_image = im.fromarray(image, 'RGB')
input_image.show()


def convolution(kernel, input_region):

    for riga in range(3):
        for colonna in range(3):
            convolution = np.zeros(3)
            for valoriRGB in range(3):
                convolution[valoriRGB] += kernel[riga][colonna]*input_region[riga][colonna][valoriRGB]

    output_values = np.array(convolution.clip(0, 255), dtype=np.uint8)
    return output_values


def convol_layer(image, kernel):
    res=[]
    for i in range (len(image)-2):
        for j in range (len(image)-2):
            res.append(convolution(kernel, image[i:i+3,j:j+3]))
    return res

for i in range(13):
    print(i)
    res=convol_layer(image,kernel)
    res=np.array(res).reshape(len(image)-2,len(image)-2,3)
    image=res
    im.fromarray(res.astype('uint8'), 'RGB').show()


im.fromarray(res.astype('uint8'), 'RGB').show()


