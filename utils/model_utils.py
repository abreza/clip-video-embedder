import math

def count_parameters(model, verbose=True):

    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable

def compute_tensor_size(tensor_shape):
    size = math.prod(tensor_shape) * 4 / 1024/1024

    if size/1024 <1:
      print(f"Size: {size:.2f} MB")
    else:
      print(f"Size: {size/1024:.2f} GB")