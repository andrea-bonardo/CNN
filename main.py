import numpy as np
from PIL import Image as im
from keras.datasets import cifar10
import random
import time



class NeuronLayer():
    def __init__(self, numero_di_kernel):
        
        scritto=False

        with open("kernels.txt", "r") as f:
            if f.read() != "":
                scritto=True

        with open("kernels.txt", "a") as f:
            if scritto==False:
                for i in range(numero_di_kernel):
                    for j in range(9):
                        f.write(f"{(random.randint(51,151)-100)/100} ")
                    f.write("\n")
        f.close()  

        self.kernel = np.loadtxt("kernels.txt",dtype=float,skiprows=0,max_rows = numero_di_kernel)

numero_di_kernel=10

kernel=NeuronLayer(numero_di_kernel)

kernel = kernel.reshape(numero_di_kernel,3,3)

print(kernel.shape)

[training_set_inputs, training_set_outputs],[x_test, y_test] = cifar10.load_data()


image = np.asarray(x_test[0], dtype=np.uint8)
input_image = im.fromarray(image, 'RGB')
#input_image.show()
print(image.shape)


def convolution(kernel, input_region):
    return np.sum(np.multiply(kernel, input_region))


def convol_layer(image, kernel):
    res=[]
    for i in range (len(image)-2):
        for j in range (len(image)-2):
            res.append(convolution(kernel, image[i:i+3,j:j+3]))

    return np.array(res).reshape(len(image)-2,len(image)-2)


conv=[]

for i in range(numero_di_kernel):
    conv.append(convol_layer(image, kernel[i]))

conv = np.array(conv)

print(conv.shape)



