""" [AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
## More on Computational Graphs
Conceptually, autograd keeps a record of data (tensors) and all executed
operations (along with the resulting new tensors) in a directed acyclic
graph (DAG) consisting of Function objects. In this DAG, leaves are the
input tensors, roots are the output tensors. By tracing this graph from
roots to leaves, you can automatically compute the gradients using the
chain rule.

#### In a forward pass, autograd does two things simultaneously:

- run the requested operation to compute a resulting tensor
- maintain the operation’s gradient function in the DAG.

#### The backward pass kicks off when `.backward()` is called on the DAG root. `autograd` then:

- computes the gradients from each `.grad_fn`,
- accumulates them in the respective tensor’s .grad attribute
- using the chain rule, propagates all the way to the leaf tensors.
"""
import torch

# INIT value
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True) # or w.requires_grad_(True)
b = torch.randn(3, requires_grad=True)

def one_forward_then_backward(x, y, w, b,
                             TRACK_GRAD = True):
    '''
    1. Only obtain the `grad` properties which have `requires_grad = True`
    2. Perform gradients calculation by `backward()` once on a graph. If we need to 
       do several `backward` calls on the same graph, set `retain_graph=True`.
    '''
    print('\n## Forward:')
    z = torch.matmul(x, w) + b if TRACK_GRAD else (torch.matmul(x, w) + b).detach()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print('Gradient function for z =', z.grad_fn)
    print('Gradient function for loss =', loss.grad_fn)


    print('\n## Backward: compute gradients by calling `loss.backward()`')
    loss.backward()
    print(w.grad)
    print(b.grad)


def check_reqires_grad_in_forward(x, w, b):
    '''
    All tensors with `requires_grad=True` are tracking and support gradient computation.
    If we don't need to do that, for example, only want to do forward computations in trained model.
    Use `torch.no_grad()` block:
    '''
    print('\n## Check reqires_grad in forward:')
    z = torch.matmul(x, w)+b
    print('Is tracking ?', z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w)+b
    print('Is tracking ?', z.requires_grad)
    # or, use `detach`
    z = torch.matmul(x, w)+b
    print('Is tracking ?', z.detach().requires_grad)

    print('''
    When you want to disable gradient tracking:
    1. To mark some parameters in your neural network at frozen parameters. This is a very common scenario for finetuning a pretrained network.
    2. To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.
    ''')


if __name__ == '__main__':

    one_forward_then_backward(x, y, w, b)

    check_reqires_grad_in_forward(x, w, b)

    try:
        print('\n## If not tracking gradient')
        one_forward_then_backward(x, y, w, b, TRACK_GRAD=False)
    except Exception as error:
        print("ERROR:", type(error), error)




    #NOTE: Jacobian Product: matmul(v, Jacobian matrix)????
    # inp = torch.eye(5, requires_grad=True)
    # v = torch.ones_like(inp)

    # inp = torch.ones(5, requires_grad=True)
    # v = torch.ones_like(inp)

    # print(inp, inp.size(), '\n',v)
    # out = (inp+2).pow(2)  # (x + 2)^2

    # out.backward(v, retain_graph=True)
    # print("First call\n", inp.grad)
    # out.backward(v, retain_graph=True)
    # print("\nSecond call\n", inp.grad)
    # inp.grad.zero_()
    # out.backward(v, retain_graph=True)
    # print("\nCall after zeroing gradients\n", inp.grad)