def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	total_count = 0
	for layer in layers:
		if layer['type']=='dense':
			count=layer['input_size']*layer['output_size']
			if 'bias' not in layer or layer['bias']:
				count+=layer['output_size']
			total_count+=count
		else:
			count=layer['in_channels']*layer['out_channels']*layer['kernel_size']*layer['kernel_size']
			if 'bias' not in layer or layer['bias']:
				count+=layer['out_channels']
			total_count+=count
	return total_count
