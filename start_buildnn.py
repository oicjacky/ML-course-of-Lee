# [build neural network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(DEVICE))


class NeuralNetwork(nn.Module):
    """ Every module in PyTorch subclasses the `nn.Module`.
    A neural network is a module itself that consists of other modules(layers).
    
    To use the model, call `NeuralNetwork(data)`:
    it executes the `NeuralNetwork.forward`, but do not call NeuralNetwork().forward() directly !
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def take_more_look_into_neural_network():
    input_image = torch.rand(3, 28, 28)
    print("3 images of size 28x28:", input_image.size())


    print("`nn.Flatten` convert 28x28 image into 784 pixel values")
    flatten = nn.Flatten(start_dim = 1, end_dim = -1)
    flat_image = flatten(input_image)
    print("dimension index from 1 to -1", flat_image.size())


    print("`nn.Linear` applies a linear transformation on the input")
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())


    print("`nn.ReLu`:= max(0, x), applied after linear transformations to introduce nonlinearity")
    print(f"Before ReLU: {hidden1[:,0:5]}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1[:,0:5]}\n\n")


    print("""`nn.Sequential` is an ordered container of modules. 
    Doing:
        nn.Flatten(1, -1),
        nn.Linear(28*28, 20),
        nn.ReLU(),
        nn.Linear(20, 10)\n""")
    seq_modules = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(28*28, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    logits = seq_modules(input_image)
    print(input_image.size(), "becomes", logits.size())


    print("`nn.Softmax` to scale the output values of last hidden layer into [0,1] along dimension index `dim=1`")
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    print(pred_probab.size(), "with `dim=1` values sum to 1,", torch.sum(pred_probab, dim=1))


if __name__ == '__main__':

    print("\n\n## Define the Class")
    model = NeuralNetwork().to(DEVICE)
    print(model)

    X = torch.rand(1, 28, 28, device=DEVICE)
    #NOTE: model(X) == model.__call__(X)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class index: {y_pred} with prob {pred_probab}")


    print("\n\n## Model Layers")
    take_more_look_into_neural_network()


    print("\n\n## Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    