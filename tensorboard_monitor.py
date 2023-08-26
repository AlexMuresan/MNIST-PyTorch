import numpy as np

from load_data import MNIST
from train_net import StupidNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    writer = SummaryWriter()
    model = StupidNet(28, 28)

    train_images_file = "data/train-images-idx3-ubyte"
    train_labels_file = "data/train-labels-idx1-ubyte"

    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    train_dataset = MNIST(train_images_file, train_labels_file, transforms=mnist_transforms, 
                          set_float=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    images, labels = next(iter(train_dataloader))

    grid = make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()

    for n_iter in range(10):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)