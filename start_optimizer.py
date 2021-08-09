# """ [Optimizing Model Parameter](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

# Modeling Architect
# ==================
#     1. you have a ML problem and given data
#     2. build appropriate model
#     3. train, validate and test the model by
#        optimizing its parameters on data
#         - In each epoch,
#           a. the model makes a guess about output
#           b. calculates the error (loss)
#           c. collect the derivatives of the loss function w.r.t parameters
#           d. optimzies these parameters using gradient descent

# """
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from start_buildnn import NeuralNetwork


def train_loop(dataloader: DataLoader, model: nn.Module,
               loss_fn, optimizer: torch.optim.Optimizer
               ):
    size = len(dataloader.dataset)
    for batch_index, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            loss, current = loss.item(), batch_index * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # import pdb
            # pdb.set_trace()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    print("\n## Prepare data and build model")
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    model = NeuralNetwork()

    print("\n## Hyperparameters: Epochs, Batch size, Learning rate")
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    print("\n## Loss function")
    loss_fn = nn.CrossEntropyLoss()

    print("""\n## Optimizer e.g. SGD, ADAM, RMSProp
    1. `optimizer.zero_grad()` reset the gradients of model parameters. Note that gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
    2. Backpropagate the loss with a call to `loss.backward()`
    3. Call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.
    """)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


    for ep in range(epochs):
        print(f"Epoch {ep+1}\n---------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")