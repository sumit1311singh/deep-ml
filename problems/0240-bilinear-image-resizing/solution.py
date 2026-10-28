import numpy as np

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize an image using bilinear interpolation.

    Args:
        image: 2D (grayscale) or 3D (RGB) array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image

    Returns:
        Resized image as a nested list with values rounded to 2 decimals
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape[:2]

    # Determine scaling factors
    scale_y = H / new_height
    scale_x = W / new_width

    # Allocate output array, preserving the input dimensions
    if image.ndim == 2:
        output = np.zeros((new_height, new_width), dtype=float)
    else:
        output = np.zeros(
            (new_height, new_width, image.shape[2]), dtype=float
        )

    for i in range(new_height):
        for j in range(new_width):
            # Map output coordinates to source coordinates
            y = i * scale_y
            x = j * scale_x

            # Find neighbouring pixels
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)

            # Calculate interpolation weights
            dy = y - y0
            dx = x - x0

            # Retrieve neighbouring pixel values
            top_left = image[y0, x0]
            top_right = image[y0, x1]
            bottom_left = image[y1, x0]
            bottom_right = image[y1, x1]

            # Bilinear interpolation (works for grayscale and RGB)
            output[i, j] = (
                (1 - dx) * (1 - dy) * top_left
                + dx * (1 - dy) * top_right
                + (1 - dx) * dy * bottom_left
                + dx * dy * bottom_right
            )

    # Return a nested list rounded to 2 decimal places
    return np.round(output, 2).tolist()