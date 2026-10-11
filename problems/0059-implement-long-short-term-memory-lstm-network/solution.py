import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		ht = []
		current_hidden_state = initial_hidden_state
		current_cell_state = initial_cell_state
		for t in range(len(x)):
			xt = np.expand_dims(x[t], axis=1)
			zt = np.concatenate([current_hidden_state, xt], axis=0)

			ft = sigmoid(self.Wf @ zt + self.bf)
			it = sigmoid(self.Wi @ zt + self.bi)
			cct = np.tanh(self.Wc @ zt + self.bc)
			ot = sigmoid(self.Wo @ zt + self.bo)

			current_cell_state = ft * current_cell_state + it * cct
			current_hidden_state = ot * np.tanh(current_cell_state)

			ht.append(current_hidden_state)

		return np.array(ht), current_hidden_state, current_cell_state



