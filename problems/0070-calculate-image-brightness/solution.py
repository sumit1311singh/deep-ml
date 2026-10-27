import torch

def calculate_brightness(img) -> float:
    """
    Calculate the average brightness of a grayscale image using PyTorch.

    Args:
        img: A 2D list where each element represents a pixel value between 0-255.

    Returns:
        The average brightness rounded to two decimal places, or -1 for invalid inputs.
    """
    # Write your code here
    if not img or not img[0]:
        return -1

    if any(len(row) != len(img[0]) for row in img):
        return -1

    arr = torch.tensor(img, dtype=torch.float32)

    if torch.any((arr<0) | (arr>255)):
        return -1

    return round(torch.mean(arr).item(), 2)