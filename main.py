import numpy as np
from PIL import Image as im
from keras.datasets import cifar10
import random
import time

class NeuronLayer():
    def __init__(self, numero_prec, numero_di_kernel, profondita):
        # with open("kernels.txt", "a") as f:
        #     for i in range(numero_di_kernel):
        #         for j in range(9*profondita):
        #             f.write(f"{(random.randint(51,151)-100)/100} ")
        #         f.write("\n")
        # f.close()
        self.kernel = np.loadtxt("kernels.txt", dtype=float, skiprows=numero_prec, max_rows=numero_di_kernel)
        self.kernel = self.kernel.reshape(numero_di_kernel, profondita, 3, 3)

    def __getitem__(self, index):
        return self.kernel[index]

    def __len__(self):
        return len(self.kernel)
    
class dense_layer():
    def __init__(self, input, output):
        #with open("weight.txt", "a") as f:
        #    for i in range(input):
        #        for j in range(output):
        #             f.write(f"{(random.randint(51,151)-100)/100} ")
        #        f.write("\n")

        #    for i in range(output):
        #        f.write(f"{(random.randint(51,151)-100)/100} ")
        #f.close()

        self.synaptic_weights = np.loadtxt("weight.txt", dtype=float, max_rows=input)
        self.bias = np.loadtxt("weight.txt", dtype=float, skiprows=input)



class CNN():
    def __init__(self, layer1, layer2, layer3, layer4, fc_layer):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.fc_layer = fc_layer

    def convolution(self, kernel, input_region):
        return np.sum(np.multiply(kernel, input_region))

    def convol_layer(self, image, kernel):
        input_channels, height, width = image.shape
        output_height = height - 2
        output_width = width - 2
        output = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                patch = image[:, i:i+3, j:j+3]
                output[i, j] = self.convolution(kernel, patch)
        return output

    def max_pooling(self, image, size):
        channels, height, width = image.shape
        pooled_height = height // size
        pooled_width = width // size
        pooled = np.zeros((channels, pooled_height, pooled_width))
        for c in range(channels):
            for i in range(0, height, size):
                for j in range(0, width, size):
                    pooled[c, i//size, j//size] = np.max(image[c, i:i+size, j:j+size])
        return pooled
    
    def __softmax(self, x):
        exp = np.exp(x - np.max(x))  
        return exp / exp.sum(axis=0)
    





    def think(self, image):
        image = np.transpose(image, (2, 0, 1))  # Shape: (3, 32, 32)
        print("Input shape:", image.shape)

        # Layer 1
        conv = []
        for kernel in self.layer1:
            conv.append(self.convol_layer(image, kernel))
        conv = np.array(conv)  # Shape: (32, 30, 30)
        print("After Layer1:", conv.shape)

        # Layer 2
        conv2 = []
        for kernel in self.layer2:
            conv2.append(self.convol_layer(conv, kernel))
        conv2 = np.array(conv2)  # Shape: (64, 28, 28)
        print("After Layer2:", conv2.shape)

        # Max Pooling
        conv2 = self.max_pooling(conv2, 2)  # Shape: (64, 14, 14)
        print("After Pooling1:", conv2.shape)

        # Layer 3
        conv3 = []
        for kernel in self.layer3:
            conv3.append(self.convol_layer(conv2, kernel))
        conv3 = np.array(conv3)  # Shape: (128, 12, 12)
        print("After Layer3:", conv3.shape)

        # Layer 4
        conv4 = []
        for kernel in self.layer4:
            conv4.append(self.convol_layer(conv3, kernel))
        conv4 = np.array(conv4)  # Shape: (256, 10, 10)
        print("After Layer4:", conv4.shape)

        # Max Pooling
        conv4 = self.max_pooling(conv4, 2)  # Shape: (256, 5, 5)
        print("After Pooling2:", conv4.shape)


        flattened = conv4.flatten()
        print("Flattened shape:", flattened.shape)  # Should be (6400,)

        output = np.dot(flattened, self.fc_layer.synaptic_weights) + self.fc_layer.bias
        print(output.shape)  


        probabilities = self.__softmax(output)
        return probabilities


if __name__ == "__main__":
    # Initialize layers with correct numero_prec (cumulative kernels from previous layers)
    layer1 = NeuronLayer(0, 32, 3)          # 32 kernels, depth=3 (RGB)
    layer2 = NeuronLayer(32, 64, 32)        # 64 kernels, depth=32 (output from layer1)
    layer3 = NeuronLayer(32+64, 128, 64)    # 128 kernels, depth=64 (output from layer2)
    layer4 = NeuronLayer(32+64+128, 256, 128) # 256 kernels, depth=128 (output from layer3)
    fc_layer = dense_layer(6400, 10)     # 10 output neurons

    print(fc_layer.synaptic_weights.shape)
    print(fc_layer.bias.shape)

    (_, _), (x_test, _) = cifar10.load_data()
    image = x_test[0]  # Shape: (32, 32, 3)

    cnn = CNN(layer1, layer2, layer3, layer4, fc_layer)
    output = cnn.think(image)
    print(output)
