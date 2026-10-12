import numpy as np

class SimpleRNN:
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Initializes the RNN with random weights and zero biases.
		"""
		self.hidden_size = hidden_size
		self.W_xh = np.random.randn(hidden_size, input_size)*0.01
		self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
		self.W_hy = np.random.randn(output_size, hidden_size)*0.01
		self.b_h = np.zeros((hidden_size, 1))
		self.b_y = np.zeros((output_size, 1))
		
	def forward(self, x):
		"""
		Forward pass through the RNN for a given sequence of inputs.
		"""
		current_hidden_state = np.zeros((self.hidden_size, 1))

		self.output_states = []
		self.hidden_states = []

		for t in range(len(x)):
			new_hidden_state = np.tanh(self.W_xh @ x[t].reshape(-1, 1) + self.W_hh @ current_hidden_state + self.b_h)
			new_output_state = self.W_hy @ new_hidden_state + self.b_y

			current_hidden_state = new_hidden_state
			self.hidden_states.append(new_hidden_state)
			self.output_states.append(new_output_state)

		return np.array(self.output_states)

	def backward(self, x, y, learning_rate):
		"""
		Backpropagation through time to adjust weights based on error gradient.
		"""
		dh_next = np.zeros((self.hidden_size, 1))
		dW_hy, db_y = np.zeros(self.W_hy.shape), np.zeros(self.b_y.shape)
		dW_xh, dW_hh, db_h = np.zeros(self.W_xh.shape), np.zeros(self.W_hh.shape), np.zeros(self.b_h.shape)

		for t in reversed(range(len(y))):
			dy = self.output_states[t] - y[t].reshape(-1, 1)

			dW_hy += dy @ self.hidden_states[t].T
			db_y += dy

			dh_t = self.W_hy.T @ dy + self.W_hh.T @ dh_next
			dh_raw = dh_t * (1 - self.hidden_states[t]**2)
			
			xt = np.array(x[t]).reshape(1, -1)
			dW_xh += dh_raw @ xt
			
			ht_prev = self.hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, 1))
			dW_hh += dh_raw @ ht_prev.T
			
			db_h += dh_raw
			dh_next = dh_raw

		self.W_xh -= learning_rate * dW_xh
		self.W_hh -= learning_rate * dW_hh
		self.W_hy -= learning_rate * dW_hy
		self.b_h -= learning_rate * db_h
		self.b_y -= learning_rate * db_y
			




