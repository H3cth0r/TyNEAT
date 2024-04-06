from Tinygrad import Tensor
class Node:
    def __init__(self, children, weights, response, bias, 
                 activation, aggregation, name=None, leaves=None):
        self.children           = children
        self.leaves             = leaves
        self.weights            = weights
        self.response           = response
        self.bias               = bias
        self.activation, self.activation_name = activation, activation
        self.aggregation, self.aggregation_name= aggregation, aggregation
        self.aggregation_name   = aggregation
        self.name               = name
        if leaves is not None: assert isinstance(leaves. dict)
        self.leaves             = leaves
        self.activs             = None
        self.is_reset           = None
    def __repr__(self):
        header = "Node({}, response={}, bias={}, activation={}, aggregation={})".format(
                self.name, self.response, self.bias, self.activation_name, self.aggregation_name,
        )
        child_reprs = []
        for w, child in zip(self.weights, self.children):child_reprs.append("    <- {} * ".format(w) + repr(child).replace("\n", "\n    "))
        return header + "\n" + "\n".join(child_reprs)
    def activate(self, xs, shape): 
        if not xs: return Tensor.full(shape, self.bias)
        inputs = [w * x for w, x in zip(self.weights, xs)]
        try:
            pre_activs = self.aggregation(inputs)
            activs = self.activation(self.response * pre_activs + self.bias)
            assert activs.shape == shape, "Wrong shape for node {}".format(self.name)
        except Exception:raise Exception("Failed to activate node {}".format(self.name))
        return activs
    def get_activs(self, shape):
        if self.activs is None:
            xs = [child.get_activs(shape) for child in self.children]
            self.activs = self.activate(xs, shape)
        return self.activs
    def __call__(self, **inputs):
        assert self.leaves is not None
        assert inputs
        shape = list(inputs.values())[0].shape
        self.reset()
        for name in self.leaves.keys():
            assert(
                    inputs[name].shape == shape
            ), "Wrong activs shape for leaf {}, {} != {}".format(
                    name, inputs[name].shape, shape
            )
            self.leaves[name].set_activs(inputs[name])
        return self.get_activs(shape)
    def _prereset(self):
        if self.is_reset is None:
            self.is_reset = False
            for child in self.children:child._prereset()
    def _postreset(self):
        if self.is_reset is not None:
            self.is_reset = None
            for child in self.children: child._postreset()
    def _reset(self):
        if not self.is_reset:
            self.is_reset = True
            self.activs = None
            for child in self.children: child._reset()
    def reset(self):
        self._prereset()
        self._reset()
        self._postreset()
