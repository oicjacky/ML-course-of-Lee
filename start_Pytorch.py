import torch
import torch.nn as nn
import torch.optim
import numpy as np

x_numpy = np.array([0.1, 0.2, 0.3])
x_torch = torch.tensor([0.1, 0.2, 0.3])
print(f"x_numpy = {x_numpy};\n x_tensor = {x_torch}\n")


## pytorch to and from numpy
print(f"""
from numpy to torch: {torch.from_numpy(x_numpy)},
from torch to numpy: {x_torch.numpy()}""")


## basic operations 
y_numpy = np.array([3, 4, 5])
y_torch = torch.tensor([3, 4, 5])
print(f"x + y = {x_numpy + y_numpy} and {x_torch + y_torch}")

# many functions that are in numpy are also in pytorch
print(f"norm = {np.linalg.norm(x_numpy)} , {torch.norm(x_torch)}")


# to apply an operation along a dimension,
# we use the dim keyword argument instead of axis
print("mean along the 0th dimension")
x_numpy = np.array([
    [1,2],
    [3,4.]
])
x_torch = torch.tensor([[1,2],[3,4.]])
print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))


## Tensor.view() can reshape tensors similarly to numpy.reshape()
# for example "MNIST" data,
N, C, W, H = 10000, 3, 28, 28 # number, channel, width, hight
X = torch.randn((N, C, W, H))

print(X.shape)
print(X.view(N, C, 784).shape)
print(X.view(-1, C, 784).shape)


## Broadcasting semantics
# 1. Each tensor has at least 1-dim.
# 2. When iterating over a tensor, starting at the
#    trail dimension. 
#    Note: the dim-size must be equal, one of them is 1,
#          or one of them does not exist.
x = torch.empty(5, 1, 4, 1)
y = torch.empty(   3, 1, 1)
# RuntimeError: y = torch.empty(   3, 2, 1)
print((x+y).size())


## Computation graphs
# `tensor` create a computation graph in the background,
# which write a mathematical expression as a graph
# Consider `e = (a + b) * (b + 1) with a = 2, b = 1` 
a = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(1.0, requires_grad = True)
c = a + b
d = b + 1
e = c * d
print('c', c)
print('d', d)
print('e', e) # PyTorch kept track of the computation graph for us.


## Cuda semantics
# tensor with cpu or gpu
cpu = torch.device('cpu')
gpu = torch.device('cuda') if torch.cuda.is_available() else None

x = torch.rand(10)
print("default", x)
x = x.to(gpu)
print("gpu", x)
x = x.to(cpu)
print("cpu", x)
print("Old, no recommended:\n", x.cuda(), '\n', x.cpu())


## PyTorch as an Auto Grad framework
# Consider f(x) = (x - 2)^2
# To compute f'(x) and calculate f'(1),
# use `backward()` call on the leaf variable `y`, then compute all the gradients of `y` at once.
def f(x):
    return (x - 2) **2

def fp(x):
    return 2* (x - 2)

x = torch.tensor([3.0], requires_grad=True)
y = f(x)

print(x, y)
y.backward()

print(f"""Manually calculate f\'(x) = {fp(x)};\n
PyTorch backward of y (i.e. f\'(x)) = {x.grad}""")

# Consider g(W) = 2* w1w2 + w2 cos(w1), W = (w1, w2)
# compute gradient of g and calculate the value of [pi, 1]
def g(w):
    return 2* w[0]* w[1] + w[1]* torch.cos(w[0])

def grad_g(w):
    return torch.tensor([ 2*w[1] - w[1]*torch.sin(w[0]), 2*w[0] + torch.cos(w[0]) ])

W = torch.tensor([np.pi, 1], requires_grad= True)
z = g(W)

print(W, z)
z.backward()

print(f"""Manually calculate grad g(W) = {grad_g(W)};
PyTorch backward of z (grad g(W)) = {W.grad}""")


## Using the gradients find minimum of f(x) = (x - 2)^2
# Manually, the minimum is f(2) = 0.
x = torch.tensor([5.0], requires_grad= True)
step_size = 0.25

print('iter,\t x,\t f(x),\t f\'(x),\t f\'(x) pytorch')
for i in range(15):
    y = f(x)
    y.backward()

    # torch.Tensor([1.132]).item()  Get a Python scalars if Tensor is 1-dim
    print('{},\t{:.3f},\t{:.3f},\t{:.3f},\t{:.3f}'.format(
            i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))

    x.data = x.data - step_size * x.grad # perform the update of gradient descent

    # We need to zero the grad variable since the backward()
    # call accumulates the gradients in .grad instead of overwriting.
    # The detach_() is for efficiency. You do not need to worry too much about it.
    x.grad.detach_()
    x.grad.zero_()


## Linear Regression using gradient descent
d = 2
n = 50
X = torch.randn(n, d)
true_w = torch.tensor([[-1.0], [2.0]])
y = X @ true_w + 0.1* torch.randn(n, 1)  # N(0, 0.1^2)

print('X_shape', X.shape)
print('true_w_shape', true_w.shape)
print('y_shape', y.shape)


# define a linear model with unknow function f
def f(X, w):
    return X @ w

# loss function: RSS (the residual sum of squares)
def RSS(y, y_hat):
    return torch.norm(y - y_hat)**2 / n

# analytical expression for the gradient
def grad_rss(X, y, w):
    return -2*X.t() @ (y - X @ w) / n

w = torch.tensor([[1.], [0]], requires_grad=True)
y_hat = f(X, w)

loss = RSS(y, y_hat)
loss.backward()

print('Analytical gradient', grad_rss(X, y, w).detach().view(2).numpy())
print('PyTorch\'s gradient', w.grad.view(2).numpy())

step_size = 0.1

print('iter, \tloss, \tw, \tgrad_rss, \tw.grad')
for i in range(20):
    y_hat = f(X, w)
    loss = RSS(y, y_hat)
    
    loss.backward() # compute the gradient of the loss
    
    w.data = w.data - step_size * w.grad # do a gradient descent step

    print('{},\t{:.2f},\t{},\t{},\t{}'.format(
        i, loss.item(), w.view(2).detach().numpy(), grad_rss(X, y, w).detach().view(2).numpy(), w.grad.view(2).numpy()))
    
    # We need to zero the grad variable since the backward()
    # call accumulates the gradients in .grad instead of overwriting.
    # The detach_() is for efficiency. You do not need to worry too much about it.
    w.grad.detach()
    w.grad.zero_()

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', w.view(2).detach().numpy())


## `torch.nn.Module`
# `Module` performs operation on tensors. 
# All modules are callable and can 
# be composed to create complex function. For example,
# Linear Module:
# the basic module that does a linear transformation with a intercept.
d_in = 3
d_out = 4
linear_module = nn.Linear(d_in, d_out)

example_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float)
transformed = linear_module(example_tensor)
print('example_tensor', example_tensor.shape)
print('transormed', transformed.shape)
print()
print('We can see that the weights exist in the background\n')
print('W:', linear_module.weight)
print('b:', linear_module.bias) #[x for x in linear_module.parameters() ]


## Activation functions
# such as `ReLU`, `Tanh`, `Sigmoid`
activation_fn = nn.ReLU() # instantiate an instance of ReLU
example_tensor = torch.tensor([[-1, 2, 4], [-21, -3, -4]], dtype=torch.float)
activated = activation_fn(example_tensor)
print('example_tensor', example_tensor)
print('activated', activated)


## Sequential
# `torch.nn.Sequential` provides a good interface for composing simple modules.
d_in = 3
d_hidden = 4
d_out = 1
model = nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.Tanh(),
    nn.Linear(d_hidden, d_out),
    nn.Sigmoid(),
)
example_tensor = torch.tensor([[3, 1, 2], [4, 6, 5]], dtype=torch.float)
transformed = model(example_tensor)
print('example_tensor', example_tensor)
print('transformed', transformed.detach())

# Note: to access all of parameter of `nn.Module` with `parameters()`
# here, 4 parameters in total are 2 pairs (weight, bias) of nn.Linear.
params = model.parameters()
for param in params:
    print(type(param), param.size())


## Loss function
# such as `MSELoss` and `CrossEntropyLoss`
mse_loss = nn.MSELoss()

input = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float)
target = torch.tensor([[1, 2, 2], [0, 0, 0]], dtype=torch.float)

print(mse_loss(input, target))  # (1 + 4 + 4 + 1) /6


## torch.optim
# take model parameter and learning rate as argument.
# Usage: call `backward()` to compute the gradients. 
# Note: call the `optim.zero_grad()` before `backward()` since
# by default PyTorch does and inplace add to the `.grad` member variable rather than overwriting it.
# It does both `detach_()` and `zero_()` calls on all tensor's variables.
model = nn.Linear(1, 1)
optim = torch.optim.SGD(model.parameters(), lr = 1e-2)
mse_loss = nn.MSELoss()

X_simple = torch.tensor([[1.]])
y_simple = torch.tensor([[1.]])

y_hat = model(X_simple)
print("model params before:", model.weight, model.bias)
print("model params's grad before:", model.weight.grad, model.bias.grad)
loss = mse_loss(y_hat, y_simple)

optim.zero_grad()
loss.backward()
optim.step() # w.data = w.data - lr * w.grad
print("model params after:", model.weight, model.bias)
print("model params's grad after:", model.weight.grad, model.bias.grad) # -0.1234 -  (-0.2907* 1e-2 ), 0.9781 - (-0.2907* 1e-2 )


## Linear regression using GD with automatically computed derivatives and PyTorch's Modules
step_size = 0.1
d = 2
n = 50
X = torch.randn(n, d)
true_w = torch.tensor([[-1.0], [2.0]])
y = X @ true_w + 0.1* torch.randn(n, 1)  # N(0, 0.1^2)

print('X_shape', X.shape)
print('true_w_shape', true_w.shape)
print('y_shape', y.shape)

linear_module = nn.Linear(d, 1, bias=False)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(linear_module.parameters(), lr = step_size)

print("iter, \t loss, \t w")
for i in range(20):
    loss = loss_func(linear_module(X), y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    print('{}, \t{:.2f}, \t{}'.format(
        i, loss.item(), linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', linear_module.weight.view(2).detach().numpy())


## Linnear regression using SGD
# In the previous examples, we computed the average gradient over the entire dataset (Gradient Descent). 
# We can implement Stochastic Gradient Descent with a simple modification.
step_size = 0.01

linear_module = nn.Linear(d, 1)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(linear_module.parameters(), lr=step_size)
print('iter,\tloss,\tw')
for i in range(200):
    rand_idx = np.random.choice(n) # take a random point from the dataset
    x = X[rand_idx] 
    y_hat = linear_module(x)
    loss = loss_func(y_hat, y[rand_idx]) # only compute the loss on the single point
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if i % 20 == 0:
        print('{},\t{:.2f},\t{}'.format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))

print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', linear_module.weight.view(2).detach().numpy())


## Neural Network Basics in PyTorch
# %matplotlib inline
import matplotlib.pyplot as plt

d_in = 1
X = torch.rand(200, d_in)
y = 4* torch.sin(np.pi * X) * torch.cos(6 * np.pi * X**2)

plt.scatter(X.numpy(), y.numpy())
plt.title("plot of $f(x) = 4\sin(\pi x) \cos(6\pi x^2)$")
plt.xlabel("$X$")
plt.ylabel("$y$")
plt.show()

n_hidden_1 = 32
n_hidden_2 = 32
d_out = 1
step_size = 0.05
n_epochs = 6000

neural_network = nn.Sequential(
    nn.Linear(d_in, n_hidden_1),  # 1x32
    nn.Tanh(),
    nn.Linear(n_hidden_1, n_hidden_2), # 32x32
    nn.Tanh(),
    nn.Linear(n_hidden_2, d_out),
)
loss_func = nn.MSELoss()

optim = torch.optim.SGD(neural_network.parameters(), lr = step_size)
print("iter, \t loss")
for i in range(n_epochs+1):
    y_hat = neural_network(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % (n_epochs //10) == 0:
        print(f"{i}, \t {loss.item():.5f}")


X_gridpoint = torch.from_numpy(np.linspace(0,1, 100)).float().view(-1, d_in)
y_hat = neural_network(X_gridpoint)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X_gridpoint.detach().numpy(), y_hat.detach().numpy(), 'r')
plt.title('plot of $f(x)$ and $\hat{f}(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()


## Things that might help: [Why Momentum works in Optimization](https://distill.pub/2017/momentum/)
# here only change the `momentum`:
momentum = 0.9
neural_network = nn.Sequential(
    nn.Linear(d_in, n_hidden_1),  # 1x32
    nn.Tanh(),
    nn.Linear(n_hidden_1, n_hidden_2), # 32x32
    nn.Tanh(),
    nn.Linear(n_hidden_2, d_out),
)
loss_func = nn.MSELoss()

optim = torch.optim.SGD(neural_network.parameters(), lr = step_size, momentum = momentum)
print("iter, \t loss")
for i in range(n_epochs+1):
    y_hat = neural_network(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % (n_epochs //10) == 0:  # complete every 10% procedure will print
        print(f"{i}, \t {loss.item():.5f}")


X_gridpoint = torch.from_numpy(np.linspace(0,1, 100)).float().view(-1, d_in)
y_hat = neural_network(X_gridpoint)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X_gridpoint.detach().numpy(), y_hat.detach().numpy(), 'r')
plt.title('plot of $f(x)$ and $\hat{f}(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()


## CrossEntropy Loss
# Args: (output of network, true label)
# output of network: (N_samples, C_classes) with values correspond to 
#   raw scores for each class. `CrossEntropyLoss` will does the softmax calculation.
# true label: (N_samples) with the `correct` class labels that 
#   represents [0, 1, 2, ..., C-1].
loss = nn.CrossEntropyLoss()

Inputs = [
    torch.tensor([[-1, 1], [-1, 1], [1, -1]], dtype=torch.float),
    # raw scores correspond to the correct class
    torch.tensor([[-3, 3], [-3, 3], [3, -3]], dtype=torch.float),
    # raw scores correspond to the correct class with higher confidence
    torch.tensor([[1, -1], [1, -1], [-1, 1]], dtype=torch.float),
    # raw scores correspond to the incorrect class
    torch.tensor([[3, -3], [3, -3], [-3, 3]], dtype=torch.float)
    # raw scores correspond to the incorrect class with incorrectly placed confidence
]
target = torch.tensor([1, 1, 0])

for input in Inputs:
    output = loss(input, target)
    print(input, '\n loss = ', output)


## Learning rate schedulers
# Often we don't use a fixed learning rate throught training.
# PyTorch offers learning rate schedulers to change the learning rate
# over time. Common strategies is multiplying the `lr` by a constant every epoch (e.g. 0.9)
# and halving the learning rate when the training loss flattens out.
# See [learning rate schedulers docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).


## [Convolutions](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv#torch.nn.Conv2d)
# `torch.nn.Conv2d`: the input (N, C_in, H_in, W_in) where 
# N = batch size, C_in = number of channels the image has,
# H_in & W_in = the height & width of the image.
# Modify the convolution with:
# 1. kernel size  2. stride  3. padding  [ref. CNN explainer](https://poloclub.github.io/cnn-explainer/)
image = np.array([0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.3803922 , 0.37647063, 0.3019608 ,0.46274513, 0.2392157 , 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.3529412 , 0.5411765 , 0.9215687 ,0.9215687 , 0.9215687 , 0.9215687 , 0.9215687 , 0.9215687 ,0.9843138 , 0.9843138 , 0.9725491 , 0.9960785 , 0.9607844 ,0.9215687 , 0.74509805, 0.08235294, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.54901963,0.9843138 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.7411765 , 0.09019608, 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.8862746 , 0.9960785 , 0.81568635,0.7803922 , 0.7803922 , 0.7803922 , 0.7803922 , 0.54509807,0.2392157 , 0.2392157 , 0.2392157 , 0.2392157 , 0.2392157 ,0.5019608 , 0.8705883 , 0.9960785 , 0.9960785 , 0.7411765 ,0.08235294, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.14901961, 0.32156864, 0.0509804 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.13333334,0.8352942 , 0.9960785 , 0.9960785 , 0.45098042, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.32941177, 0.9960785 ,0.9960785 , 0.9176471 , 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.32941177, 0.9960785 , 0.9960785 , 0.9176471 ,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.4156863 , 0.6156863 ,0.9960785 , 0.9960785 , 0.95294124, 0.20000002, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.09803922, 0.45882356, 0.8941177 , 0.8941177 ,0.8941177 , 0.9921569 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.94117653, 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.26666668, 0.4666667 , 0.86274517,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.5568628 ,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.14509805, 0.73333335,0.9921569 , 0.9960785 , 0.9960785 , 0.9960785 , 0.8745099 ,0.8078432 , 0.8078432 , 0.29411766, 0.26666668, 0.8431373 ,0.9960785 , 0.9960785 , 0.45882356, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.4431373 , 0.8588236 , 0.9960785 , 0.9490197 , 0.89019614,0.45098042, 0.34901962, 0.12156864, 0., 0.,0., 0., 0.7843138 , 0.9960785 , 0.9450981 ,0.16078432, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.6627451 , 0.9960785 ,0.6901961 , 0.24313727, 0., 0., 0.,0., 0., 0., 0., 0.18823531,0.9058824 , 0.9960785 , 0.9176471 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.07058824, 0.48627454, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.32941177, 0.9960785 , 0.9960785 ,0.6509804 , 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.54509807, 0.9960785 , 0.9333334 , 0.22352943, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.8235295 , 0.9803922 , 0.9960785 ,0.65882355, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.9490197 , 0.9960785 , 0.93725497, 0.22352943, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.34901962, 0.9843138 , 0.9450981 ,0.3372549 , 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.01960784,0.8078432 , 0.96470594, 0.6156863 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.01568628, 0.45882356, 0.27058825,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.], dtype=np.float32)
image_torch = torch.from_numpy(image).view(1, 1, 28, 28)

# a gaussian blur kernel
gaussian_kernel = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

conv = nn.Conv2d(1, 1, 3)
# manually set the conv weight
conv.weight.data[:] = gaussian_kernel
convolved = conv(image_torch)

print(f"before conv: {image_torch.shape}, after conv: {convolved.shape}")
plt.title('original image')
plt.imshow(image_torch.view(28,28).detach().numpy())
plt.show()

plt.title('blurred image')
plt.imshow(convolved.view(26,26).detach().numpy())
plt.show()

# take in an RGB (3 channel) and output a 16 channel image.
image_channels = 3  # RGB images
out_channels = 16  # a hyperparameter 
kernel_size = 3  # also a hyperparameter
batch_size = 3
image_width = 32
image_height = 32

image = torch.randn(batch_size, image_channels, image_width, image_height)
m = nn.Conv2d(image_channels, out_channels, kernel_size)
convolved = m(image)

print(f"`nn.Conv2d (kernel_size = 3)`'s weight: (in_channel, out_channel) = (1, 1) is {conv.weight.shape};\n (in_channel, out_channel) = (3, 16) is  {m.weight.shape}")
print('image shape', image.shape)
print('convolved image shape', convolved.shape)


## Dataset and DataLoader
# Dataset class:
# Your custom dataset should inherit `Dataset` and overwrite
# - __len__ : provide the size of a dataset.
# - __getitem__ : support the indexing with i-th sample representing `dataset[i]`.
from torch.utils.data import Dataset, DataLoader

class FakeDataset(Dataset):

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y 
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> tuple:
        return self.x[index], self.y[index]

# DataLoader class: provide an iterator to perform
# - Batching data
# - Shuffling data
# - Load the data in parallel using `multiprocessing` workers
# Another parameter of interest is `collate_fn` which define the samples being
# batched using it.
x = np.random.rand(100, 10)
y = np.random.rand(100)

dataset = FakeDataset(x, y)
dataloader = DataLoader(dataset, batch_size = 4, 
                        shuffle = True, num_workers = 0) 

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched)


## Mixed Presision Training
# [Ref: apex ](https://github.com/NVIDIA/apex)
# using mixed precision to train your networks can be
# - 2 to 4 times faster
# - memory-efficient 
# - only 3 lines of Python code

# ```
# from apex import amp
# # Declare model and optimizer as usual, with default (FP32) precision
# model = torch.nn.Linear(10, 100).cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# # Allow Amp to perform casts as required by the opt_level
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
# ...
# # loss.backward() becomes:
# with amp.scale_loss(loss, optimizer) as scaled_loss:
#     scaled_loss.backward()
# ...
# ```


