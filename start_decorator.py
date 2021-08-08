# [Decorator](https://hackmd.io/@rh0jTfFDTO6SteMDq91tgg/HkFHtDjDO)
"""
Usage
=====
def wrapper(function_to_be_changed):
    def some_process(*args):
        print("\tmore operations starts!")
        print("\tinput args:", type(args), args)
        output = function_to_be_changed(*args)
        print("\tmore operations ends!")  
        return output
    return some_process

@wrapper
def add(x, y):
    return x + y
"""
# function
def add(x, y):
    " a simple function"
    return x + y

def add_more(x, y, *args):
    " `*args` collects other positional arguments as tuple "
    print(f"[add_more] x = {x}, y = {y}, args = {args}")
    res = x + y
    for ele in args:
        res += ele
    return res

# decorator
def new_function(function_to_be_changed, *args):
    """ different from `wrapper` decorator method
    you should call `new_function` for following usage.
    """
    print("\tmore operations starts!") 
    print("\tinput args:", type(args), args)
    output = function_to_be_changed(*args)
    print("\tmore operations ends!")   
    return output 

def wrapper(function_to_be_changed):
    """ a decorator that wrap `function_to_be_changed` to
    do more things
    """
    def some_process(*args):
        print("\tmore operations starts!")
        print("\tinput args:", type(args), args)
        output = function_to_be_changed(*args)
        print("\tmore operations ends!")  
        return output
    return some_process 


if __name__ == '__main__':

    # args = x, y
    x = 1
    y = 3

    print("Simple function:")
    print(add(x, y))

    print("New function call:")
    print(new_function(add, x, y))

    print("Wrapper it, then call origin function:")
    add = wrapper(add)
    print(add(x, y))

    
    print('--------------------------------------')
    print("Case2: more arguments")
    print(add_more(x, y, 100, 200))

    print("New function call:")
    print(new_function(add_more, x, y, 100, 200))

    print("Wrapper it, then call origin function:")
    add_more = wrapper(add_more)
    print(add_more(x, y, 100, 200))