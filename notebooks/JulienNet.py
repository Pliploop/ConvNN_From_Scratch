import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from sklearn.model_selection import train_test_split


class SetGenerator():
    def __init__(self, train_batch_size, test_batch_size) -> None:
        self.basic_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.val_size = 0.1

        self.training_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                         download=True, transform=self.basic_transforms)

        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=self.basic_transforms)

        self.classes = ['plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        train_indices, val_indices, _, _ = train_test_split(
            range(len(self.training_set)),
            self.training_set.targets,
            stratify=self.training_set.targets,
            test_size=self.val_size,
        )

        self.train_split = torch.utils.data.Subset(
            self.training_set, train_indices)
        self.val_split = torch.utils.data.Subset(
            self.training_set, val_indices)

        self.training_data_loader = torch.utils.data.DataLoader(self.train_split, batch_size=self.train_batch_size,
                                                                shuffle=True, num_workers=2)
        self.val_data_loader = torch.utils.data.DataLoader(self.val_split, batch_size=self.train_batch_size,
                                                                shuffle=True, num_workers=2)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.test_batch_size,
                                                            shuffle=False, num_workers=2)


    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        return np.transpose(npimg, (1, 2, 0))

    def show_random(self):
        dataiter = iter(self.training_data_loader)
        images, labels = dataiter.next()
        image = self.imshow(torchvision.utils.make_grid(images))
        return image


def block(x):
    x = nn.Conv2d(3, 64, kernel_size=3, padding=1)(x)
    x = F.relu(x)
    x = nn.MaxPool2d(2, 2)(x)
    x = nn.Conv2d(64, 64, kernel_size=3, padding=1)(x)
    x = nn.BatchNorm2d(64)(x)
    x = F.relu(x)
    x = nn.MaxPool2d(2, 2)(x)
    return x


class JulienNet(nn.Module):

    def __init__(self,dropout) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.mp_1 = nn.MaxPool2d(2, 2)
        self.drop_1_2d = nn.Dropout2d(p=0.2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.mp_2 = nn.MaxPool2d(2, 2)
        self.drop_2_2d = nn.Dropout2d(p=0.3)


        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d(2, 2)
        self.drop_3_2d = nn.Dropout2d(p=0.4)

        self.fc_1 = nn.Linear(2048, 128)
        self.dropout_1 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(128, 64)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax()

        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn_1(x)
        x = self.mp_1(x)
        if self.dropout:
            x = self.drop_1_2d(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn_2(x)
        x = self.mp_2(x)
        if self.dropout:
            x = self.drop_2_2d(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.bn_3(x)
        x = self.mp_3(x)
        if self.dropout:
            x = self.drop_3_2d(x)

        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc_1(x))
        if self.dropout:
            x = self.dropout_1(x)
        x = F.relu(self.fc_2(x))
        if self.dropout:
            x = self.dropout_2(x)
        x = F.relu(self.fc_3(x))
        x = self.softmax(x)
        return x


class JNetTrainer:
    def __init__(self, SetGenerator, n_epochs,dropout) -> None:
        self.SetGenerator = SetGenerator
        self.net = JulienNet(dropout=dropout)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)
        self.n_epochs = n_epochs
        self.train_loss_list = np.zeros(n_epochs)
        self.val_loss_list = np.zeros(n_epochs)
        self.train_accuracy_list = np.zeros(n_epochs)
        self.val_accuracy_list = np.zeros(n_epochs)
        self.train_loss_list[:] = np.NaN
        self.val_loss_list[:] = np.NaN
        self.train_accuracy_list[:] = np.NaN
        self.val_accuracy_list[:] = np.NaN
        self.fig,self.ax=plt.subplots(1,2)
        
    def train(self):
        epochs = trange(self.n_epochs, leave=True)
        for epoch in epochs:  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.SetGenerator.training_data_loader, 0):

                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                # print(torch.argmax(outputs,dim=1))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    epochs.set_description('[{}, {}], loss: {}\%'.format(
                        epoch + 1, 100*(i + 1)/len(self.SetGenerator.train_split), running_loss/10), refresh=True)
                    running_loss = 0.0
            # print(outputs.shape)

            # getting accuracy for training and test

            train_total = 0
            val_total = 0
            train_correct = 0
            val_correct = 0
            

            
            with torch.no_grad():
                self.net.eval()
                for data in self.SetGenerator.training_data_loader:

                    images, labels = data[0].to(
                    self.device), data[1].to(self.device)

                    outputs = self.net(images)
                    predicted=torch.argmax(outputs, dim = 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                self.train_accuracy_list[epoch] = 100 * train_correct / train_total
                self.train_loss_list[epoch] = self.criterion(outputs, labels)
                
                for data in self.SetGenerator.val_data_loader:

                    images, labels = data[0].to(
                    self.device), data[1].to(self.device)

                    outputs = self.net(images)
                    predicted=torch.argmax(outputs, dim = 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                self.val_accuracy_list[epoch] = 100 * val_correct / val_total
                self.val_loss_list[epoch] = self.criterion(outputs, labels)
                self.net.train()   




        print('Finished Training')
        PATH='./Jnet.pth'
        torch.save(self.net.state_dict(), PATH)

    def plot_metrics(self):
        ax=self.ax
        self.fig.clear()
        
        ax[0].plot(self.train_loss_list,color='k')
        ax[0].plot(self.val_loss_list,color='g')
        ax[0].set(title="Loss",xlabel="n_epochs",ylabel="loss")
        
        ax[1].plot(self.train_accuracy_list,color='b')
        ax[1].plot(self.val_accuracy_list,color='r')
        ax[1].set(title="Accuracy",xlabel="n_epochs",ylabel="accuracy",ylim=[0,100])

        plt.show()