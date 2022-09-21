import torch

class MultModel(torch.nn.Module):
    def __init__(self, val):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = val

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a * x

def model_fn(model_dir):
    model = MultModel(2)
    return model
