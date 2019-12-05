#!/usr/bin/env python
import time
import numpy as np
import torch
import torchvision
from torchmps import MPS
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['text.usetex'] = True

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Miscellaneous initialization
torch.manual_seed(0)

# functions to show an image
def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# functions to load data
def load_data(dataset):
    # Get the training and test sets
    if  dataset == 'cifar':
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    else:
        transform = transforms.ToTensor()
    print("=====================================")
    print("The dataset we are training:", dataset)
    if  dataset == 'mnist':
        train_set = datasets.MNIST('./mnist', download=True, transform=transform)
        test_set = datasets.MNIST('./mnist', download=True, transform=transform, train=False)
    elif dataset == 'fashionmnist':
        train_set = datasets.FashionMNIST('./fashionmnist', download=True, transform=transform)
        test_set = datasets.FashionMNIST('./fashionmnist', download=True, transform=transform, train=False)
    elif dataset == 'cifar':
        train_set = datasets.CIFAR10('./CIFAR10', download=True, transform=transform)
        test_set = datasets.CIFAR10('./CIFAR10', download=True, transform=transform, train=False)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # print(len(train_set))
    dim = len(train_set[1][0][0])
    num_channel = len(train_set[1][0])

    # Put MNIST data into dataloaders
    samplers = {'train': torch.utils.data.SubsetRandomSampler(range(num_train)),
                'test': torch.utils.data.SubsetRandomSampler(range(num_test))}
    loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
               sampler=samplers[name], drop_last=True) for (name, dataset) in 
               [('train', train_set), ('test', test_set)]}
    num_batches = {name: total_num // batch_size for (name, total_num) in
                   [('train', num_train), ('test', num_test)]}

    print(f"Training on {num_train} images \n"
          f"(testing on {num_test}) for {num_epochs} epochs")
    print(f"Maximum MPS bond dimension = {bond_dim}")
    print(f" * {'Adaptive' if adaptive_mode else 'Fixed'} bond dimensions")
    print(f" * {'Periodic' if periodic_bc else 'Open'} boundary conditions")
    print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
    if l2_reg > 0:
        print(f" * L2 regularization = {l2_reg:.2e}")
    print()
#     print(np.size(train_set[1]))
          
    return samplers, loaders, num_batches, dim, num_channel
          
# Let's start training!
def mps_train(loaders, num_channel):
    # Initialize the MPS module
    mps = MPS(input_dim=dim**2, output_dim=10, bond_dim=bond_dim, 
              adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)

    # Set our loss function and optimizer
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mps.parameters(), lr=learn_rate, 
                                 weight_decay=l2_reg)

    if num_channel == 3:
        size = [num_channel, batch_size, dim**2]
    else:
        size = [batch_size, dim**2]
          
          
    train_acc = []
    test_acc  = []
    ave_loss = []
    run_time = []
          
    for epoch_num in range(1, num_epochs+1):
        running_loss = 0.
        running_acc = 0.

        for inputs, labels in loaders['train']:
            inputs, labels = inputs.view(size), labels.data

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)

            # Compute the loss and accuracy, add them to the running totals
            loss = loss_fun(scores, labels)
            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item() / batch_size
                running_loss += loss
                running_acc += accuracy

            # Backpropagate and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"### Epoch {epoch_num} ###")
        print(f"Average loss:           {running_loss / num_batches['train']:.4f}")
        train_acc = train_acc + [ running_acc / num_batches['train'] ]
        print(f"Average train accuracy: {running_acc / num_batches['train']:.4f}")
        ave_loss = ave_loss + [ running_loss / num_batches['train'] ]

        # Evaluate accuracy of MPS classifier on the test set
        with torch.no_grad():
            running_acc = 0.
              
            for inputs, labels in loaders['test']:
                inputs, labels = inputs.view(size), labels.data

                # Call our MPS to get logit scores and predictions
                scores = mps(inputs)
                _, preds = torch.max(scores, 1)
                running_acc += torch.sum(preds == labels).item() / batch_size

        print(f"Test accuracy:          {running_acc / num_batches['test']:.4f}")
        test_acc = test_acc + [ running_acc / num_batches['test'] ]
        print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
        run_time = run_time + [ int(time.time()-start_time) ]
#         print(test_acc)
              
    return run_time, ave_loss, train_acc, test_acc


#####################################
# MPS parameters
bond_dim      = 20
adaptive_mode = False
periodic_bc   = False

# Training parameters
num_train  = 2000
num_test   = 1000
batch_size = 10
num_epochs = 40
learn_rate = 1e-4
l2_reg     = 0.

# choose dataset
# data    = 'mnist'
# data    = 'fashionmnist'
# data    = 'cifar'

TRAIN_ACC = []
TEST_ACC  = []
AVE_LOSS = []
RUN_TIME = []

for data in ['cifar', 'mnist', 'fashionmnist']:
    start_time = time.time()
    samplers, loaders, num_batches, dim, num_channel = load_data(data)

    # visualization
    # get some random training images
    dataiter = iter(loaders['train'])
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # training
    run_time, ave_loss, train_acc, test_acc = mps_train(loaders, num_channel)
    TRAIN_ACC = TRAIN_ACC + [train_acc]
    TEST_ACC  = TEST_ACC  + [test_acc]
    AVE_LOSS  = AVE_LOSS  + [ave_loss]
    RUN_TIME  = RUN_TIME  + [run_time]


## Visualization
epoch_num = [range(len(i)) for i in RUN_TIME]
fig, (ax0, ax1) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6))
ax0.plot(epoch_num[0], AVE_LOSS[0], 'bo-')
ax0.plot(epoch_num[1], AVE_LOSS[1], 'ro-')
ax1.plot(epoch_num[0], TRAIN_ACC[0], 'bo-')
ax1.plot(epoch_num[1], TRAIN_ACC[1], 'ro-')
ax1.plot(epoch_num[0], TEST_ACC[0], 'bo--')
ax1.plot(epoch_num[1], TEST_ACC[1], 'ro--')
ax1.set_xlabel(r'Number of epochs',fontsize=16)
ax0.set_ylabel('Average Loss', fontname='Times', fontsize=16)
ax1.set_ylabel('Accuracy',fontsize=16)
ax0.legend(('Mnist','Fashionmnist'), loc='upper right')
ax1.legend(('Mnist(train)','Fashionmnist(train)','Mnist(test)','Fashionmnist(test)',), loc='lower right')
fig.savefig('test.png', transparent=True, dpi=400)


