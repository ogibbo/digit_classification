import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Defining the model
model = nn.Sequential(
    nn.Linear(28*28,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(64,10)
)

# Choosing optimizer
optimiser = optim.SGD(model.parameters(),lr = 0.005)

# Defining a loss for classification task
loss = nn.CrossEntropyLoss() 

# Importing the dataset
train_data = datasets.MNIST('data',train = True, download = True, transform=transforms.ToTensor())
train,val = random_split(train_data,[55000,5000])
train_loader = DataLoader(train,batch_size= 32)
val_loader = DataLoader(val,batch_size = 32)

# Training and validation loop

no_epochs = 7

for epoch in range(no_epochs): # Each epoch is a full pass of dataset
    
    losses = list()
    accuracy_list = list()
    model.train()
    
    for batch in train_loader:
        
        # x is an image and y is the label
        x, y = batch
        b = x.size(0) # Batch size
        x = x.view(b,-1) # x has b rows where each row has length 28*28

        # Forward pass
        l = model(x)

        # Compute objective function
        J = loss(l,y)

        # Cleaning the gradient
        model.zero_grad()

        # Want to compute partial derivatives of J wrt parameters
        J.backward()

        # Step in the opposite direction (ie using learning rate and partial derivs)
        optimiser.step()

        # Keeping track of losses and accuracies
        losses.append(J.item())
        accuracy_list.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch+1}, train loss: {torch.tensor(losses).mean():0.2f}, training accuracy: {torch.tensor(accuracy_list).mean():0.2f}')
    
    losses = list()
    accuracy_list = list()
    model.eval()

    for batch in val_loader:
        # x is an image and y is the label
        x, y = batch
        b = x.size(0) # batch size
        x = x.view(b,-1) # x has b rows where each row has length 28*28

        # Forward (Here we do not want to compute gradients)
        with torch.no_grad():
            l = model(x)

        # Compute objective function
        J = loss(l,y)

        losses.append(J.item())
        accuracy_list.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch+1}, validation loss: {torch.tensor(losses).mean():0.2f}, validation accuracy: {torch.tensor(accuracy_list).mean():0.2f}')

# Saving the model
torch.save(model,'model_weights.pth')