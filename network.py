# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.preprocessing import normalize
from torch.nn.functional import softmax

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        print("Start to train weights...")
        num_data =  len(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)

        print("Weights initialized")
        print(W.shape)
        
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = np.float32(train_data[i] - rho)
            W += np.outer(t, t)
        
        print("Hebb rule step done")

        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
    
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def _run(self, init_s):
        if self.asyn==False:
            """
            Synchronous update
            """
            # Compute initial state energy
            s = init_s

            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
        else:
            """
            Asynchronous update
            """
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                for j in range(100):
                    # Select random neuron
                    idx = np.random.randint(0, self.num_neuron) 
                    # Update s
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
    
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()

class ContinuousHopfieldNetwork(object):
    '''
    Simple implementation of a continuous Hopfield network (described by Ramsauer et al.)

    Data images first need to be flattened for training and prediction.
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes


    def train(self, train_data: np.ndarray):
        '''
        For classification, train_data is (z_i, t_i) where t_i is the classification vector
        '''
        self._stored_patterns = train_data.T

    def energy(self, state):
        pass
    
    def predict(self, test_data, beta=1, num_iter=1, hide_output=False):
        state = np.copy(test_data)
        labels = np.zeros((len(test_data), self.num_classes))
        state = np.concatenate((state, labels), axis=1)
        for i in tqdm(range(len(test_data)), disable=hide_output):
            for j in range(num_iter):
                matmu = self._stored_patterns.T[:, :-self.num_classes] @ state[i][:-self.num_classes].T
                matmu *= beta

                soft = np.exp(matmu)/np.sum(np.exp(matmu))

                state[i] = self._stored_patterns @ soft
                labels[i] = state[i][-self.num_classes:]
        return state, labels



'''

Store [[1, 2, 3, 4],
       [2, 3, 4, 5]]

       ->

       [[1, 2],
        [2, 3],
        [3, 4],
        [4, 5]]

Predict on [[3, 4, 5, 6]] -> [[3],
                              [4],
                              [5],
                              [6]]

beta * (X.T @ S) = [[3 + 8 + 15 + 24],
                   [6 + 12 + 20 + 30]]
                 = [[50],
                    [68]]

softmax(beta * (X.T @ S)) = [[exp(50)/(exp(50) + exp(68))],
                             [exp(68)/(exp(50) + exp(68))]]
                          = [[1.523e-8],
                             [0.999]]

X @ softmax(beta * (X.T @ S)) = [[~2],
                                 [~3],
                                 [~4],
                                 [~5]]

'''
if __name__ == "__main__":
    chn = ContinuousHopfieldNetwork(num_classes=3)
    train_signal = np.array([[1,2,3,4],
                             [7,1,3,1],
                             [10,9,8,7]])
    
    train_signal = normalize(train_signal, axis=1)

    labels = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]])
    
    train_data = np.concatenate((train_signal, labels), axis=1)
    #train_data = train_signal
    
    test_data = normalize(np.array([[1, 2, 3, 4]]), axis=1)

    chn.train(train_data)
    prediction = chn.predict(test_data, beta=400, num_iter=1)
    print("Prediction: ", prediction)


'''
[1 * 0 + 2 * 1, 2 * 0 + 3 * 1
'''