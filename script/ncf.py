import os
import urllib
import zipfile
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class NNHybridFiltering(nn.Module):

    def __init__(self, n_users, n_items, n_genres, embdim_users, embdim_items, embdim_genres, n_activations, rating_range):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=n_users,embedding_dim=embdim_users)
        self.item_embeddings = nn.Embedding(num_embeddings=n_items,embedding_dim=embdim_items)
        self.genre_fc = nn.Linear(n_genres, embdim_genres)

        self.fc1 = nn.Linear(embdim_users+embdim_items+embdim_genres,n_activations)
        self.fc2 = nn.Linear(n_activations,1)
        self.rating_range = rating_range

    def forward(self, X):
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:,0])
        # print(embedded_users.shape)
        embedded_items = self.item_embeddings(X[:,1])
        # print(embedded_items.shape)
        embedded_genres = self.genre_fc(X[:,2:].float())

        # print(embedded_genres.shape)

        # Concatenate user, item and genre embeddings
        embeddings = torch.cat([embedded_users,embedded_items,embedded_genres],dim=1)
        # print(embeddings.shape)
        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)
        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(preds) * (self.rating_range[1]-self.rating_range[0]) + self.rating_range[0]
        return preds

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=5, scheduler=None):
    model = model.to(device) # Send model to GPU if available
    since = time.time()

    costpaths = {'train':[],'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (inputs,labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) * labels.size(0)

            # Step along learning rate scheduler when in train
            if (phase == 'train') and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            costpaths[phase].append(epoch_loss)
            print('{} loss: {:.4f}'.format(phase, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return costpaths


# Download the data from the GroupLens website
def load_data():
    datapath = './data/'

    # Load data
    ratings = pd.read_csv(os.path.join(datapath,'df_filtered_data.csv'))
    ratings = pd.merge(ratings, onehot_encode_map, on='anime_id', how='left')
    ratings.drop(['watching_status','watched_episodes'],axis=1,inplace=True)

    onehot_encode_map = pd.read_csv(os.path.join(datapath,'genres_vectorized.csv'))
    return ratings, onehot_encode_map

def split_data(ratings):
    # extract data features and labels
    x_col = ratings.columns
    x_col = x_col.drop('rating')
    X = ratings.loc[:,x_col]
    y = ratings.loc[:,'rating']

    # Split our data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.2)
    # Split our training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,random_state=0, test_size=0.2)

    return X_train, X_val, X_test, y_train, y_val, y_test


def prep_dataloaders(X_train,y_train,X_val,y_val,batch_size):
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(),
                            torch.from_numpy(np.array(y_train)).float())
    valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(),
                            torch.from_numpy(np.array(y_val)).float())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

def predict_rating(model, userId, animeId, genre, device):

    # Get predicted rating
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        user_anime_tensor = torch.tensor([userId, animeId])
        genre_tensor = torch.tensor(genre.values[0])

        concat_tensor = torch.cat((user_anime_tensor, genre_tensor))

        X = concat_tensor.long().view(1,-1)
        X = X.to(device)

        pred = model.forward(X)
        return pred


def generate_recommendations(animes,X,model,userId,device):
    # Get predicted ratings for every movie
    pred_ratings = []
    for anime in animes['MAL_ID'].tolist():
        genre = animes.loc[(animes['anime_id'] == anime), animes.columns[8:]]
        pred = predict_rating(model,userId,anime,genre, device)
        pred_ratings.append(pred.detach().cpu().item())
    # Sort animes by predicted rating
    idxs = np.argsort(np.array(pred_ratings))[::-1]
    recs = animes.iloc[idxs]['MAL_ID'].values.tolist()
    # Filter out animes already watched by user
    animes_watched = X.loc[X['user_id']==userId, 'anime_id'].tolist()
    recs = [rec for rec in recs if not rec in animes_watched]
    # Filter to top 10 recommendations
    recs = recs[:10]
    # Convert movieIDs to titles
    recs_names = []
    print(recs)
    for rec in recs:
        recs_names.append(animes.loc[animes['MAL_ID']==rec, 'Name'].values[0])
    return recs_names

if __name__ == '__main__':
    # Load data
    ratings, onehot_encode_map = load_data()
    
    # Get the predicted rating for a random user-item pair
    rating = predict_rating(model,userId=34,animeId=10,genre=X_test.loc[(X_test['user_id'] == 260755) & (X_test['anime_id'] == 5204), X_test.columns[2:]], device=device)
    print('Predicted rating is {:.1f}'.format(rating.detach().cpu().item()))

    batchsize = 64
    trainloader,valloader = prep_dataloaders(X_train,y_train,X_val,y_val,batchsize)

    # Train the model
    dataloaders = {'train':trainloader, 'val':valloader}
    n_users = X.loc[:,'user_id'].max()+1
    n_items = X.loc[:,'anime_id'].max()+1
    n_genres = 47

    model = NNHybridFiltering(n_users,
                        n_items,
                        n_genres,
                        embdim_users=50,
                        embdim_items=50,
                        embdim_genres=25,
                        n_activations = 100,
                        rating_range=[1.,10.])
    criterion = nn.MSELoss()
    lr=0.001
    n_epochs=2
    wd=1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cost_paths = train_model(model,criterion,optimizer,dataloaders, device,n_epochs, scheduler=None)
