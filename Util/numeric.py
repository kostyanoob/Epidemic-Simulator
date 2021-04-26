import math
from typing import Tuple, List, Union, Iterable
import numpy as np


def floatify_string(string):
    try:
        return float(string)
    except ValueError:
        return 0.0


def modified_sigmoid(x: float, coefficient: float, shift: float) -> float:
    """
    return as single value of a modified sigmoid function, which correpsonds to an
    argument x.
    :param x: argument to the sigmoid function
    :param coefficient: a scaling coefficient
    :param shift: a shifting term
    :return: a value of a modified sigmoid function
    """
    return 1/(1+math.exp(-coefficient*(x-shift)))


def modified_sigmoid_vector(x_list: Iterable[Union[float, int]],
                            coefficient: Union[float, int],
                            shift: Union[float, int]) -> List[float]:
    """
    returns a modified (parameterized) sigmoid function values list
    corresponding to the argument list "x_list
    :param x_list: list of arguments (floats)
    :param shift: parameter determining the shift of the sigmoid function
                  (i.e. the value at which the sigmoid crosses the value 0.5)
    :param coefficient: a scaling factor determining the slope of the sigmoid
    :return: list of sigmoid function values corresponding to the x_list argument values.
    """
    a = []
    for x in x_list:
        a.append(modified_sigmoid(coefficient, shift))
    return a