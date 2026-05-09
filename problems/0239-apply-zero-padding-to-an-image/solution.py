import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or numpy array of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """

    if pad_width < 0:
        return -1

    img = np.array(img, dtype=int)
    if img.ndim!=2:
        return -1
    elif img.shape[0]==0 or img.shape[1]==0:
        return -1
    
    padded = np.pad(img, pad_width, mode='constant', constant_values=0)
    
    return padded.tolist()  