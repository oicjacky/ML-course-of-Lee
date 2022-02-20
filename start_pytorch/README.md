# PyTorch Introduction
- [PyTorch Introduction](#pytorch-introduction)
  - [start_Pytorch.py](#start_pytorchpy)
  - [start_dataloader.py](#start_dataloaderpy)
  - [start_buildnn.py](#start_buildnnpy)
  - [start_autograd.py](#start_autogradpy)
  - [start_optimizer.py](#start_optimizerpy)

## start_Pytorch.py

See [Colab sample code](https://colab.research.google.com/drive/1CmkRJ5R41D2QIZXQLszMt68TWUrGd-yf) and [video]((https://www.youtube.com/watch?v=kQeezFrNoOg&feature=youtu.be)):  
- 微分梯度計算automatic differentiation
- 常用架構與函數PyTorch common functions in deep learning
- Data Processing with PyTorch `DataSet`
- Mixed Presision Training in PyTorch

## start_dataloader.py

See [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html):  
- PyTorch provides two data primitives:  
    dataset and dataloader in `torch.utils.data`.
    
- It also proide a number of pre-loaded datasets, 
  - [Image](https://pytorch.org/vision/stable/datasets.html)
  - [Text](https://pytorch.org/text/stable/datasets.html)
  - [Audio](https://pytorch.org/audio/stable/datasets.html)


## start_buildnn.py

See [Build Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html):  
- A neural network is a module itself that consists of other modules (layers).
- Every module in PyTorch subclasses the `torch.nn.Module`.


## start_autograd.py

See [Automatic Differentiation with torch.autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html):  
- Computational Graphs called directed acyclic graph (DAG).
- In a **forward** pass, autograd does two things simultaneously:
  - run the requested operation to compute a resulting tensor
  - maintain the operation’s gradient function in the DAG.
- The **backward** pass kicks off when `.backward()` is called on the DAG root. `autograd` then:
  - computes the gradients from each `.grad_fn`,
  - accumulates them in the respective tensor’s `.grad` attribute
  - using the chain rule, propagates all the way to the leaf tensors.

## start_optimizer.py

See [Optimizing Model Parameter](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html):  
- Modeling Architect
    1. you have a ML problem and given data
    2. build appropriate model
    3. train, validate and test the model by optimizing its parameters on data.  
        In each epoch,
        1. the model makes a guess about output
        2. calculates the error (loss)
        3. collect the derivatives of the loss function w.r.t parameters
        4. optimzies these parameters using gradient descent
