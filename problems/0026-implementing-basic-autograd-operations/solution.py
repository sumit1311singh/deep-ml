class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data + other.data, (self, other), '+')
		#out = self.data + other.data
		def _backward():
			self.grad += 1 * out.grad
			other.grad += 1 * out.grad  
		out._backward = _backward 
		return out

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data * other.data, (self, other), '*')
		#out = self.data * other.data
		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad 
		out._backward = _backward 
		return out

	def relu(self):
		x = self.data
		#out = 
		out = Value(x if x > 0 else 0, (self, ), 'relu')
		def _backward():
			if out.data>0:
				self.grad += 1 * out.grad
			else:
				self.grad += 0
		out._backward = _backward
		return out

	def backward(self):
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self) #start building topo sort with self
		
		self.grad = 1
		for node in reversed(topo):
			node._backward()