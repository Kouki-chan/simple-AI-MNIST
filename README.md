# Pytorch info

to change the vairbales, use 'dtype = torch.(type)'
    ex 'dtype = torch.double' or 'dtype = torch.float16'

to check the size of the tensor use 'size()' -> remember that size is a function in this case

to set the tensors to 0,1, random numbers, or an empty tensor
    use:
        'x = tenser.zeros(2,2)' for 0s
        'x = tensor.ones(2,2)' for 1s
        'x = tensor.rand(2,2)" for random numbers
        'x = tensor.empty(2,2)' for an empty tensor 
to change the dimension, change the the numbers in the parathensis
    'x = tensor.empty(x)' for a vector
    'x = tensor.empty(x,y)' for a 2d array
    'x = tensor.empty(x,y,z)' for a 3d array


to set vaules to the sensor use:
    'x = torch.tensor([2.5, 0.5])'
    remember that tensor only takes in one argument and it has to be in an array

