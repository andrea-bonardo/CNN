import numpy as np
from keras.datasets import cifar10
import random
import time
import matplotlib.pyplot as plt

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

    def __relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def __relu_derivative(self, x):
        return np.where(x > 0, 1, -0.01)
    
    def convolution(self, kernel, input_region):
        return np.sum(np.multiply(kernel, input_region))

    def convol_layer(self, image, kernel):
        input_channels, height, width = image.shape
        output = np.zeros((height-2, width-2))
        for i in range(height-2):
            for j in range(width-2):
                output[i,j] = self.convolution(kernel, image[:, i:i+3, j:j+3])
        return output

    def max_pooling(self, image, size):
        channels, height, width = image.shape
        pooled_height = height // size
        pooled_width = width // size
        pooled = np.zeros((channels, pooled_height, pooled_width))
        indices = [[[(0,0)] * pooled_width for _ in range(pooled_height)] for _ in range(channels)]
        for c in range(channels):
            for i in range(0, height, size):
                for j in range(0, width, size):
                    patch = image[c, i:i+size, j:j+size]
                    max_val = np.max(patch)
                    pooled[c, i//size, j//size] = max_val
                    max_idx = np.argmax(patch)
                    offset_i, offset_j = np.unravel_index(max_idx, patch.shape)
                    indices[c][i//size][j//size] = (i + offset_i, j + offset_j)
        return pooled, indices
    
    def __softmax(self, x):
        exp = np.exp(x - np.max(x))  
        return exp / exp.sum() + 1e-8

    def backprop(self, y, learning_rate):
        # Calcolo del gradiente della loss
        d_loss = self.probabilities - y
        d_fc_w = np.outer(self.flattened, d_loss)
        d_fc_b = d_loss

        #DENSE LAYER
        d_pool4 = np.dot(self.fc_layer.synaptic_weights, d_loss).reshape(self.conv4.shape)


        #CONV LAYER 4
        d_conv4_prepool = np.zeros_like(self.conv4pre)
        for c in range(d_conv4_prepool.shape[0]):
            for i in range(self.conv4.shape[1]):
                for j in range(self.conv4.shape[2]):
                    orig_i, orig_j = self.pool4_idx[c][i][j]
                    d_conv4_prepool[c, orig_i, orig_j] = d_pool4[c, i, j]
        d_conv4_prepool *= self.__relu_derivative(self.conv4pre)
        d_layer4 = [np.zeros_like(k) for k in self.layer4]
        for k in range(len(self.layer4)):
            for i in range(self.conv3.shape[1] - 2):
                for j in range(self.conv3.shape[2] - 2):
                    d_layer4[k] += d_conv4_prepool[k, i, j] * self.conv3[:, i:i+3, j:j+3]
            self.layer4.kernel[k] -= learning_rate * d_layer4[k] / self.conv3.shape[0]

    


        #CONV LAYER 3
        d_conv3 = np.zeros_like(self.conv3)
        for k in range(len(self.layer4)):
            kernel = self.layer4[k]
            for i in range(d_conv4_prepool.shape[1]):
                for j in range(d_conv4_prepool.shape[2]):
                    d_conv3[:, i:i+3, j:j+3] += kernel * d_conv4_prepool[k, i, j]
        d_conv3 *= self.__relu_derivative(self.conv3)
        d_layer3 = [np.zeros_like(k) for k in self.layer3]
        for k in range(len(self.layer3)):
            for i in range(self.conv2.shape[1] - 2):
                for j in range(self.conv2.shape[2] - 2):
                    d_layer3[k] += d_conv3[k, i, j] * self.conv2[:, i:i+3, j:j+3]
            self.layer3.kernel[k] -= learning_rate * d_layer3[k] / self.conv2.shape[0]



        #CONV LAYER 2
        d_conv2_postpool = np.zeros_like(self.conv2)  # Shape: come output del pooling, es. (64, 14, 14)
        for k in range(len(self.layer3)):
            kernel = self.layer3[k]
            for i in range(d_conv3.shape[1]):
                for j in range(d_conv3.shape[2]):
                    d_conv2_postpool[:, i:i+3, j:j+3] += kernel * d_conv3[k, i, j]

        clip_value = 10.0
        d_conv2_postpool = np.clip(d_conv2_postpool, -clip_value, clip_value)
        d_conv2_prepool = np.zeros_like(self.conv2pre)  # Shape: es. (64, 28, 28)
        for c in range(self.conv2pre.shape[0]):
            for i in range(self.conv2.shape[1]):  
                for j in range(self.conv2.shape[2]):
                    orig_i, orig_j = self.conv2_idx[c][i][j]
                    d_conv2_prepool[c, orig_i, orig_j] = d_conv2_postpool[c, i, j]

        d_layer2 = [np.zeros_like(k) for k in self.layer2]
        d_conv = np.zeros_like(self.conv)
        for k in range(len(self.layer2)):
            kernel = self.layer2[k]
            for i in range(d_conv2_prepool.shape[1]):
                for j in range(d_conv2_prepool.shape[2]):
                    patch = self.conv[:, i:i+3, j:j+3]
                    grad_val = d_conv2_prepool[k, i, j]
                    d_layer2[k] += grad_val * patch
            d_layer2[k] = np.clip(d_layer2[k], -clip_value, clip_value)
            self.layer2.kernel[k] -= learning_rate * d_layer2[k] / self.conv.shape[0]
        d_conv *= self.__relu_derivative(self.conv)


        #CONV LAYER 1
        d_layer1 = [np.zeros_like(k) for k in self.layer1]
        for k in range(len(self.layer1)):
            for i in range(self.input.shape[1] - 2):
                for j in range(self.input.shape[2] - 2):
                    d_layer1[k] += d_conv[k, i, j] * self.input[:, i:i+3, j:j+3]
            self.layer1.kernel[k] -= learning_rate * d_layer1[k]

        self.fc_layer.synaptic_weights -= learning_rate * d_fc_w
        self.fc_layer.bias -= learning_rate * d_fc_b


    def train(self, training_set_inputs, training_set_outputs, loss_graph, epochs=1, learning_rate=0.001, batch_size=32):
        start = time.time()
        mancante= 0
        temp = time.strftime("%H:%M:%S", time.gmtime(mancante))
        total_batches = len(training_set_inputs) // batch_size
        total_steps = epochs * total_batches * batch_size
        cont=0
            
        for epoch in range(epochs):
            for i in range(0, len(training_set_inputs), batch_size):
                batch_X = training_set_inputs[i:i+batch_size]
                batch_y = training_set_outputs[i:i+batch_size]
                batch_loss = 0
                
                
                for x, y in zip(batch_X, batch_y):
                    # Forward pass
                    prob = self.think(x)
                    
                    # Calculate loss
                    loss = -np.log(prob[np.argmax(y)])
                    batch_loss += loss
                    
                    # Backpropagation
                    self.backprop(y, learning_rate)
                    percentuale = (cont/total_steps)*100
                    print(f"{percentuale:.5f}%      t-rimanente:{temp}")
                    cont+=1

                loss_graph = np.append(loss_graph, batch_loss / len(batch_X))
                passato = time.time() - start
                mancante = passato / percentuale * (100-percentuale)
                temp = time.strftime("%D%H:%M:%S", time.gmtime(mancante))
                

                
    def think(self, image):
        self.input = np.transpose(image, (2, 0, 1))  # Shape: (3, 32, 32)

        # Layer 1
        self.conv = []
        for kernel in self.layer1:
            self.conv.append(self.convol_layer(self.input, kernel))
        self.conv = np.array(self.conv)  # Shape: (32, 30, 30)
        self.conv = self.__relu(self.conv)

        # Layer 2
        self.conv2 = []
        for kernel in self.layer2:
            self.conv2.append(self.convol_layer(self.conv, kernel))
        self.conv2 = np.array(self.conv2)  # Shape: (64, 28, 28)
        self.conv2pre = self.__relu(self.conv2)

        # Max Pooling
        self.conv2, self.conv2_idx = self.max_pooling(self.conv2pre, 2)  # Shape: (64, 14, 14)

        # Layer 3
        self.conv3 = []
        for kernel in self.layer3:
            self.conv3.append(self.convol_layer(self.conv2, kernel))
        self.conv3 = np.array(self.conv3)  # Shape: (128, 12, 12)
        self.conv3 = self.__relu(self.conv3)

        # Layer 4
        self.conv4 = []
        for kernel in self.layer4:
            self.conv4.append(self.convol_layer(self.conv3, kernel))
        self.conv4 = np.array(self.conv4)  # Shape: (256, 10, 10)
        self.conv4pre = self.__relu(self.conv4)

        # Max Pooling
        self.conv4, self.pool4_idx = self.max_pooling(self.conv4pre, 2)  # Shape: (256, 5, 5)


        self.flattened = self.conv4.flatten()  # Shape (6400,)

        output = np.dot(self.flattened, self.fc_layer.synaptic_weights) + self.fc_layer.bias

        self.probabilities = self.__softmax(output)
        return self.probabilities


if __name__ == "__main__":
    layer1 = NeuronLayer(0, 32, 3)          # 32 kernels, depth=3 (RGB)
    layer2 = NeuronLayer(32, 64, 32)        # 64 kernels, depth=32 (output from layer1)
    layer3 = NeuronLayer(32+64, 128, 64)    # 128 kernels, depth=64 (output from layer2)
    layer4 = NeuronLayer(32+64+128, 256, 128) # 256 kernels, depth=128 (output from layer3)
    fc_layer = dense_layer(6400, 10)     # 10 output neurons

    cnn = CNN(layer1, layer2, layer3, layer4, fc_layer)

    [training_set_inputs, training_set_outputs],[x_test, y_test] = cifar10.load_data()

    tr=input("training: ?")
    if tr=="s":

        training_set_inputs = np.array(training_set_inputs)
        training_set_inputs = training_set_inputs.astype(np.float32) / 255.0

        training_set_outputs = np.eye(10)[training_set_outputs.flatten()]


        loss=np.array([])
        cnn.train(training_set_inputs, training_set_outputs, loss, epochs=1, learning_rate=0.001, batch_size=32)

        plt.plot(loss)


    percentuale={}
    for i in range(10):
        percentuale[i]=[0,0]

    for i in range(len(x_test[:1000])):
        percentuale[int(y_test[i])][0]+=1
        percentuale[int(y_test[i])][1] += 1 if y_test[i]==np.argmax(cnn.think(x_test[i])) else 0
    
        
    for i in range(10):
        print(f"{i}: {percentuale[i][1]/percentuale[i][0]*100:.4f}")