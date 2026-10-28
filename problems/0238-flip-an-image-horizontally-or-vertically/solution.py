import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    # Your code here
    if not image or not image[0]:
        return -1
    
    if direction != 'horizontal' and direction != 'vertical':
        return -1

    if direction == 'vertical':
        image = np.flip(image, axis=0)
    else:
        image = np.flip(image, axis=1)

    return image