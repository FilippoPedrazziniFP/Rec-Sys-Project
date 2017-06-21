# StackedAutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
# nn.Module is the parent class in the pytorch library
# we need the methods of the parent class
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() # ineritance
        # first connection (number of features, number of neurons) INPUT --> FIRST_H_LAYER
        self.fc1 = nn.Linear(nb_movies, 20) # we are trying to detect 20 features
        self.fc2 = nn.Linear(20, 10) # second hidden layer (encoding)
        self.fc3 = nn.Linear(10, 20) # first hidden layer (decoding)
        self.fc4 = nn.Linear(20, nb_movies) # last decode
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # forward propagation 
        # x = input vector of all the movies
        # this method simulate the fact that input goes in the network forward
        # activating the different neurons in the different layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # the last is the final decode, in order to recreate the original input
        return x # vector of predicting ratings
sae = SAE()
criterion = nn.MSELoss() # to evaluate the model
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # parameters for our model



# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # to make it a float (RMSE)
    for id_user in range(nb_users):
        # it cannot accept a single vector, it needs a batch (more dimensions --> 2D) 
        input = Variable(training_set[id_user]).unsqueeze(0) # (0) index of the new dimension       
        target = input.clone() # copy of the input
        if torch.sum(target.data > 0) > 0: # all the ratings of that user, 
            # but we have just to consider the larger than 0, 
            # consideriamo l'utente solo se ha fatto il rate almeno a un item
            output = sae(input) # predictions
            target.require_grad = False # reduce the computation - 
            # it makes sure that we compute the gradient
            # not seeing the target
            output[target == 0] = 0 # they dont count in the computation
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.


print('test loss: '+str(test_loss/s))



