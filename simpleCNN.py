###############################################################################
# This is a bare bones example.  If this doesn't work, there are serious
# issues.
###############################################################################

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import os
import time
import datetime

class BasicCNN(nn.Module):
    def __init__(self, input_channels, class_cnt):
        super().__init__()
        self.class_cnt = class_cnt
        self.input_channels = input_channels
        self.net_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, class_cnt)
        )
        
    def forward(self, x):
        logits = self.net_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer, device):
    # Get number of SAMPLES and init training mode
    size = len(dataloader.dataset)
    model.train()
    
    # For each batch...
    for batch, (X, y) in enumerate(dataloader):
        # Move data to device
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Every so often, print status
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn, data_name, device):
    # Get number of samples, number of batches, and init EVALUATION mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    # Without calculating gradients...
    test_loss, correct = 0, 0
    with torch.no_grad():
        # For each batch...
        for X, y in dataloader:
            # Move data to device...
            X, y = X.to(device), y.to(device)
            
            # Predict
            pred = model(X)
            
            # Sum up loss and accurate predictions
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    # Compute average
    test_loss /= num_batches
    correct /= size
    
    # Print results
    print(data_name + f" Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Return loss just in case it's wanted for Tensorboard or somesuch
    return test_loss

def main():
    # Create data transform
    data_transform = v2.Compose([   v2.ToImage(),
                                    v2.ToDtype(torch.float32, scale=True)])
    
    # Load up CIFAR10 data
    # (This will download files to a folder "data")
    training_data = datasets.CIFAR10(root="data", train=True, download=True, 
                                     transform=data_transform)
    test_data = datasets.CIFAR10(root="data", train=False, download=True, 
                                 transform=data_transform)
    
    # Create data loaders
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # Set device
    device = "cuda"
    
    # Create network and move to device
    model = BasicCNN(input_channels=3, class_cnt=10)
    print(model)
    model = model.to(device)
    
    # Set up loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Now your training begins...
    print("Begin training...")
    start_time = time.time()
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        train_loss = test(train_dataloader, model, loss_fn, "Train", device)
        test_loss = test(test_dataloader, model, loss_fn, "Test", device)
    end_time = time.time()
    print("Done!") 
    
    elapsed = datetime.timedelta(seconds=(end_time-start_time))    
    print("Training time:", str(elapsed))
    
    
    
    # Save model
    out_dir = "models"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)        
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))  
    print("Model saved.")             
    
if __name__ == "__main__":
    main()
    