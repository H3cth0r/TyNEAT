from tinygrad import Tensor, dtypes
class AdaptiveLinearNet:
    def __init__(self, delta_w_node, input_coords, output_coords, weight_threshold=0.2, 
                 weight_max=3.0, activation="tanh", cppn_activation="identity", batch_size=1
    ):
        self.delta_w_node       = delta_w_node
        self.n_inputs           = len(input_coords)
        self.input_coords       = Tensor(input_coords, dtype=dtypes.float)
        self.n_outputs          = len(output_coords)
        self.output_coords      = Tensor(output_coords, dtype=dtypes.float)
        self.weight_threshold   = weight_threshold
        self.weight_max         = weight_max
        self.activation         = activation
        self.cppn_activation    = cppn_activation
        self.batch_size         = batch_size
        self.reset()    # TODO check what is this for
    def reset(self):
        Tensor.no_grad = True 
        self.input_to_output = {
                self.get_init_weights(self.input_coords, self.output_coords, self.delta_w_node).unsqueeze(0)
                .expand(self.batch_size, self.n_outputs, self.n_inputs)
        }
        self.w_expressed = self.input_to_output != 0
        self.batched_coords = get_coord_inputs(self.input_coords, self.output_coords, batch_size=self.batch_size)
    def get_init_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)
        n_int, n_out = len(in_coords), len(out_coords)
        zeros = Tensor.zeros((n_out, n_in), dtype=dtypes.float)
        weights = self.cppn_activation(
                w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in, pre=zeros, post=zeros, w=zeros)
        )
        clamp_weights_(weights, self.weight_threshold, self.weight_max)
        return weights
    def activate(self, inputs):
        Tensor.no_grad = True
        inputs = Tensor(inputs, dtype=dtypes.float).unsqueeze(2)
        outputs = self.activation(self.input_to_output.matmul(inputs))
        input_activs = inputs.transpose(1, 2).expand(self.batch_size, self.n_outputs, self.n_inputs)
        output_activs = outputs.expand(self.batch_size, self.n_outputs, self.n_inputs)
        (x_out, y_out), (x_in, y_in) = self.batched_coords
        delta_w = self.cppn_activation(
                self.delta_w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in, pre=input_activs, post=n_outputs, w=self.input_to_output)
        )
        self.delta_w = delta_w
        self.input_to_output[self.w_expressed] += delta_w[self.w_expressed]
        clamp_weights_(self.input_to_output, weight_threshold=0.0, weight_max=self.weight_max)
        return outputs.squeeze(2)
    @staticmethod
    def create(genome, config, input_coords, output_coords, weight_threshold=0.2, 
               weight_max=3.0, output_activation=None, activation="tanh", cppn_activation="identity", batch_size=1):
        nodes = create_cppn(genome, config, ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"], ["delta_w"], output_activation=output_activation)
        delta_w_node = nodes[0]
        return AdaptiveLinearNet(delta_w_node, input_coords, output_coords, weight_threshold=weight_threshold,
                                 weight_max=weight_max, activation=activation, cppn_activation=cppn_activation, batch_size?batch_size)
