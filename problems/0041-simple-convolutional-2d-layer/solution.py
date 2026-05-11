import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	if padding > 0:
		input_matrix = np.pad(input_matrix, padding, mode='constant', constant_values=0)

	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	out_height = (input_height - kernel_height)//stride + 1
	out_width = (input_width - kernel_width)//stride + 1

	output_matrix = np.zeros((out_height, out_width))

	windows = np.lib.stride_tricks.sliding_window_view(
    	input_matrix,
    	(kernel_height, kernel_width)
	)

	windows = windows[::stride, ::stride]

	output_matrix = np.sum(windows*kernel, axis=(-2, -1))
    
	return output_matrix
