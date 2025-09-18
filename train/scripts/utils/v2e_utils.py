import math
import torch
import numpy as np

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    rounding = 1e8 # for rounding to 8 digits
    f = (1./threshold) * math.log(threshold) # slope of linear part
    

    if isinstance(x, np.ndarray):
        # converting x into np.float64.
        if x.dtype is not np.float64:
            x = x.astype(np.float64)
        x += 1e-8 # to avoid log(0) 
        y = np.where(x <= threshold, x*f, np.log(x))
        y = np.round(y*rounding)/rounding
        y = y.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        # converting x into torch.float64.
        if x.dtype is not torch.float64:  # note float64 to get rounding to work
            x = x.double()
        x += 1e-8 # to avoid log(0) 
        y = torch.where(x <= threshold, x*f, torch.log(x))
        # important, we do a floating point round to some digits of precision
        # to avoid that adding threshold and subtracting it again results
        # in different number because first addition shoots some bits off
        # to never-never land, thus preventing the OFF events
        # that ideally follow ON events when object moves by
        y = torch.round(y*rounding)/rounding
        y = y.float()
    else:
        raise ValueError('lin_log: x must be ndarray or Tensor')
    return y