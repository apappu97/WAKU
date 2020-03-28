import numpy as np

def find_nearest_index(array, val):
    """
    Obtain the index of the element in `array` that is closest to `val`.
    
    Parameters:
    -----------
    array : `numpy.ndarray`
        The array.
    val : `float`
        The value for which to find the closest element in array.
    
    Returns:
    --------
    index : `int`
        The index of the element in array that is closest to val.
    """
    return np.abs(array - val).argmin()