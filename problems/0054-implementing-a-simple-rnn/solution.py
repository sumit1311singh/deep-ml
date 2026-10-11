import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	input_sequence, initial_hidden_state, Wx, Wh, b = np.array(input_sequence), np.array(initial_hidden_state), np.array(Wx), np.array(Wh), np.array(b) 

	hidden_state = initial_hidden_state

	for seq in input_sequence:
		hidden_state = np.tanh(Wx @ seq + Wh @ hidden_state + b)

	final_hidden_state = np.round(hidden_state, 4)

	return final_hidden_state.tolist()