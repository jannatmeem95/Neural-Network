# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math

def logistic(arr, a=1, diff=False):
    return  1/(1+np.exp(-a*arr))
        

class NeuralNetwork:
    
    mu = 0.05
    
    def train(self, sample, label, hidden_layer_config, perceptron=logistic):
        '''
        sample = numpy float matrix of samples
        label = numpy array, class 0 to n
        hidden_layer_config = list with number of perceptrons in hidden layers
        '''
        self.perceptron = perceptron
        self.mean_vect = []
        self.std_vect = []
        for i in range(sample.shape[1]):
            self.mean_vect.append(sample[ :, i].mean())
            self.std_vect.append(sample[ :, i].std())
            sample[ :, i] = (sample[ :, i] - sample[ :, i].mean())/sample[ :, i].std()
        sample = np.hstack((sample, np.ones(sample.shape[0]).reshape((sample.shape[0], 1))))
        
        label = np.eye(np.max(label)+1)[label]
        
        
        layer_config = np.array([sample.shape[1]] + hidden_layer_config + [label.shape[1]])
        self.layer_config = layer_config
        weights = []
        for i in range(len(layer_config) - 1):
            #weights.append(2 *np.random.random_sample((layer_config[i+1], layer_config[i])) - 1)
            weights.append(np.random.uniform(0,0,(layer_config[i+1], layer_config[i])) - 1)
            
        
        best_w = None
        best_error = math.inf
        
        for iter in range(1):
            count = 0
            error = 0
            del_weights = []
            for i in range(len(layer_config) - 1):
                del_weights.append(np.zeros((layer_config[i+1], layer_config[i])))
                
            for i in sample:
                v = [i]
                y = [i]
                for r in range(len(layer_config) - 1):
                    new_v = np.matmul(weights[r], y[r])
                    new_y = perceptron(new_v)
                    v.append(new_v)
                    y.append(new_y)
                        
                err = y[len(layer_config) - 1] - label[count]
                for i in err:
                    error += .5*i*i
                count += 1
                            
                del_last = err * y[-1] * (1 - y[-1])
                            
                delta = [None]*len(layer_config)
                delta[-1] = del_last

                print(del_last)

                for r in range(len(layer_config) - 1, 1, -1):
                    delta[r - 1] = np.matmul(weights[r - 1].transpose(), delta[r]) * y[r - 1] * (1 - y[r - 1])

                #print('delta:',delta)
                for r in range(len(layer_config) - 1):
                    del_weights[r] = del_weights[r] + np.matmul(delta[r + 1].reshape((delta[r+1].shape[0] , 1)), y[r].reshape((1, y[r].shape[0])))
                    
            if error < best_error:
                best_w = weights.copy()
                best_error = error
            
            for i in range(len(layer_config) - 1):
                weights[i] = weights[i] - NeuralNetwork.mu*del_weights[i]
                
        self.weights = best_w
        
        
    def test(self, features):
        for i in range(features.shape[0]):
            features[i] = (features[i] - self.mean_vect[i])/self.std_vect[i]
        features = np.append(features, [1])
        
        y = [features]
        for r in range(len(self.layer_config) - 1):
            new_v = np.matmul(self.weights[r], y[r])
            new_y = self.perceptron(new_v)
            y.append(new_y)
            
        return y[-1].argmax()
               


f = open("trainNN.txt")

train=[]
for i in f.readlines():
    train += [i.split()]
    
train = np.array(train, dtype=np.float)
train[:,-1] = train[:, -1] - np.ones(train.shape[0])

ex = train[:,:-1]
l = train[:,-1].astype('int')

nn = NeuralNetwork()
nn.train(ex, l, [5, 4])
print(nn.weights)

f = open("trainNN.txt")

test=[]
for i in f.readlines():
    test += [i.split()]
    
test = np.array(test, dtype=np.float)
test[:,-1] = test[:, -1] - np.ones(test.shape[0])

ex = test[:,:-1]
l = test[:,-1].astype('int')

error = 0
for i in range(ex.shape[0]):
    try:
        if nn.test(ex[i]) != l[i]:
            error += 1
    except IndexError:
        print(i)
        
print((1-error/ex.shape[0])*100)

