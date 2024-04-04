from tinygrad import Tensor
from functools import reduce
from operator import mul
import numpy as np
def sum_aggregation(inputs):
    if isinstance(inputs, Tensor) or isinstance(inputs, np.array):
        return inputs.sum()
    else:return inputs.sum()
def prod_aggregation(inputs): # TODO check if this will work for a numpy array
    if isinstance(inputs, Tensor):
        return reduce(Tensor.mul, inputs, 1)
    else: return reduce(mul, inputs, 1)
str_to_aggregation = {'sum': sum_aggregation, 'prod': prod_aggregation}
