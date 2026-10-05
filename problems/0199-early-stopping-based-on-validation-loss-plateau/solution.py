def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""
	# Your code here
	ans = [False] * len(val_losses)
	curr_patience = 0
	best_loss = val_losses[0]
	
	for i in range(1, len(val_losses)):
		if val_losses[i] < best_loss - min_delta:
			best_loss = val_losses[i]
			curr_patience = 0
		else:
			curr_patience += 1
			
		if curr_patience >= patience:
			ans[i] = True
			curr_patience = 0
			
	return ans
