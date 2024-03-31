from network import ContinuousHopfieldNetwork
from keras.datasets import mnist
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# prepare training data
train_X = train_X.reshape((train_X.shape[0], -1))
train_y = train_y.reshape(-1, 1)
final_train = np.concatenate((train_X, train_y), axis=1)

# prepare testing data
test_X = test_X.reshape((test_X.shape[0], -1))
classification = np.zeros(test_y.reshape(-1,1).shape)
final_test = np.concatenate((test_X, classification), axis=1)


print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
print('final_train: ' + str(final_train.shape))
print('final_test: ' + str(final_test.shape))

model = ContinuousHopfieldNetwork()

model.train(final_train)

predictions = model.predict(final_test)

print('predictions shape: ' + str(predictions.shape))

