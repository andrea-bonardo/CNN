import numpy as np
from PIL import Image as im
from keras.datasets import cifar10
import random
import time



class NeuronLayer():
    def __init__(self,numero_prec, numero_di_kernel, profondita):
        """
        with open("kernels.txt", "a") as f:
            for i in range(numero_di_kernel):
                for j in range(9*profondita):
                    f.write(f"{(random.randint(51,151)-100)/100} ")
                f.write("\n")
        f.close()  
        """
        self.kernel = np.loadtxt("kernels.txt",dtype=float,skiprows=numero_prec,max_rows = numero_di_kernel)
        self.kernel = self.kernel.reshape(numero_di_kernel,profondita,3,3)


    def __getitem__(self, index):
        return self.kernel[index]

    def __len__(self):
        return len(self.kernel)
    


    
class CNN():
    def __init__(self, layer1, layer2, layer3, layer4):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def convolution(self, kernel, input_region):
        return np.sum(np.multiply(kernel, input_region))


    def convol_layer(self, image, kernel):
        res=[]
        for i in range (len(image)-2):
            for j in range (len(image)-2):
                res.append(self.convolution(kernel, image[i:i+3,j:j+3]))

        return np.array(res).reshape(len(image)-2,len(image)-2)
    
    def max_pooling(self, image, size):
        maxed=[]
        for prof in range(len(image)):
            maxed.append(np.zeros((len(image[prof])//size,len(image[prof])//size)))
            for i in range(0,len(image[prof]),size):
                for j in range(0,len(image[prof]),size):
                    maxed[prof][i,j]=np.max(image[prof][i:i+size,j:j+size])
        
        return maxed

    def think(self, image):
        image = np.transpose(image, (2, 0, 1))
        print(image.shape)
        print(layer1.shape)
        

        conv=[]
        for i in range(len(layer1)):
            conv.append(self.convol_layer(image, layer1[i]))
        conv = np.array(conv)
        print(conv.shape)

        conv2=[]

        for i in range(len(layer2)):
            conv2.append(self.convol_layer(conv, layer2[i]))
        conv2 = np.array(conv2)
        print(conv2.shape)


        conv2=self.max_pooling(conv2,2)


        conv3=[]
        for i in range(len(layer3)):
            conv3.append(self.convol_layer(conv2, layer3[i]))
        conv3 = np.array(conv3)
        print(conv3.shape)


        conv4=[]
        for i in range(len(layer4)):
            conv4.append(self.convol_layer(conv3, layer4[i]))
        conv4 = np.array(conv4)
        print(conv4.shape)

        conv4=self.max_pooling(conv4,2)



    


if __name__ == "__main__":


    layer1=NeuronLayer(0,32,3)
    layer2=NeuronLayer(32,64,32)
    layer3=NeuronLayer(64+32,128,64)
    layer4=NeuronLayer(128+64+32,256,128)
      

    [training_set_inputs, training_set_outputs],[x_test, y_test] = cifar10.load_data()

    image = np.asarray(x_test[0], dtype=np.uint8)
    input_image = im.fromarray(image, 'RGB')
    #input_image.show()

    

    cnn = CNN(layer1, layer2, layer3, layer4)

    cnn.think(image)
    


    



