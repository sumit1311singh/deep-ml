import torch

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or torch.Tensor of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    image = torch.as_tensor(image)
    if image.ndim != 3 or image.shape[2] != 3:
        return -1

    H, W, C = image.shape

    if H==0 or W==0:
        return -1

    if torch.any((image < 0) | (image > 255)):
        return -1

    weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64)
    grayscale = torch.round(image.to(torch.float64) @ weights)

    return grayscale.to(torch.int64).tolist()