from tinygrad import Tensor, nn

class LinearRegression:
    def __init__(self, in_features: int, out_features: int):
        # TODO: store an nn.Linear submodule as self.linear
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def __call__(self, x: Tensor) -> Tensor:
        # TODO: return the output of the linear layer
        return self.linear(x)
