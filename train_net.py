import time
import torch
import subprocess

from torch import nn
from torch.optim import SGD, Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from load_data import MNIST
from neural_net import StupidNet, MidNet


def train(dataloader, model, loss_fn, optimizer, epoch, tensorboard_writer=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model.train()
    model.to(device)

    batch_loss = {}
    batch_accuracy = {}

    num_batches = len(dataloader)

    size = 0
    correct = 0
    _correct = 0
    epoch_loss = 0
    train_temp = 0

    for batch_nr, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)

        correct += _correct
        epoch_loss += loss

        batch_loss[batch_nr] = loss.item()
        batch_accuracy[batch_nr] = _correct / _batch_size

        size += _batch_size

        if batch_nr % 100 == 0:
            loss, current = loss.item(), batch_nr * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}]")

        train_temp += float(
            subprocess.getoutput(["/usr/bin/vcgencmd measure_temp"]).split("=")[1][:-2])

    correct /= size
    epoch_loss /= num_batches
    train_temp /= num_batches

    if tensorboard_writer:
        tensorboard_writer.add_scalar('Loss/train', epoch_loss, epoch)
        tensorboard_writer.add_scalar('Accuracy/train', 100*correct, epoch)
        tensorboard_writer.add_scalar('Temperature/train', train_temp, epoch)
    
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
    
    return batch_loss , batch_accuracy


def validation(dataloader, model, loss_fn, epoch, tensorboard_writer=None):
    # Total size of dataset for reference
    size = 0
    num_batches = len(dataloader)
    
    # Setting the model under evaluation mode.
    model.eval()

    test_loss, correct = 0, 0
    
    _correct = 0
    _batch_size = 0
    valid_temp = 0
    
    batch_loss = {}
    batch_accuracy = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    with torch.no_grad():
        
        # Gives X , y for each batch
        for batch , (X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            model.to(device)
            pred = model(X)
            
            batch_loss[batch] = loss_fn(pred, y).item()
            test_loss += batch_loss[batch]
            _batch_size = len(X)
            
            _correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct
            
            size+=_batch_size
            batch_accuracy[batch] = _correct/_batch_size
            valid_temp += float(
                subprocess.getoutput(["/usr/bin/vcgencmd measure_temp"]).split("=")[1][:-2])
    
    ## Calculating loss based on loss function defined
    test_loss /= num_batches
    
    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size

    valid_temp /= num_batches

    if tensorboard_writer:
        tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)
        tensorboard_writer.add_scalar('Accuracy/test', 100*correct, epoch)
        tensorboard_writer.add_scalar('Temperature/test', valid_temp, epoch)

    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    filename = str(epoch)+ "-" + time.strftime("%Y%m%d-%H%M%S") + ".pt"
    torch.save(model.state_dict(), f"./checkpoints/{filename}")
    
    return batch_loss , batch_accuracy


if __name__ == "__main__":
    train_images_file = "data/train-images-idx3-ubyte"
    train_labels_file = "data/train-labels-idx1-ubyte"

    test_images_file = "data/t10k-images-idx3-ubyte"
    test_labels_file = "data/t10k-labels-idx1-ubyte"

    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    train_dataset = MNIST(train_images_file, train_labels_file, transforms=mnist_transforms, 
                          set_float=True)
    test_dataset = MNIST(test_images_file, test_labels_file, transforms=mnist_transforms, 
                         set_float=True)

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=1)

    model = StupidNet(28, 28)
    model_CNN = MidNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=3e-3, momentum=0.9)
    optimizer_CNN = Adam(model_CNN.parameters(), lr=0.01)

    train_batch_loss = []
    train_batch_accuracy = []
    valid_batch_accuracy = []
    valid_batch_loss = []
    train_epoch_no = []
    valid_epoch_no = []

    writer = SummaryWriter(comment=f"-MidNet")

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_dataloader, model_CNN, loss_fn, 
                                                          optimizer_CNN, epoch=t, 
                                                          tensorboard_writer=writer)
        _valid_batch_loss , _valid_batch_accuracy = validation(test_dataloader, model_CNN, loss_fn, 
                                                               epoch=t, tensorboard_writer=writer)
        for i in range(len(_train_batch_loss)):
            train_batch_loss.append(_train_batch_loss[i])
            train_batch_accuracy.append(_train_batch_accuracy[i])
            train_epoch_no.append( t + float((i+1)/len(_train_batch_loss)))     
        for i in range(len(_valid_batch_loss)):
            valid_batch_loss.append(_valid_batch_loss[i])
            valid_batch_accuracy.append(_valid_batch_accuracy[i])
            valid_epoch_no.append( t + float((i+1)/len(_valid_batch_loss)))     
    print("Done!")
