from tinygrad import Tensor, dtypes
class AdaptiveNet:
    def __init__(self, w_ih_node, b_h_node, w_hh_node, b_o_node, w_ho_node, delta_w_node, 
                 input_coords, hidden_coords, output_coords, weight_threshold=0.2, activation="tanh", batch_size=1):
        self.w_ih_node  = w_ih_node
        self.b_h_node   = b_h_node
        self.w_hh_node  = w_hh_node
        self.b_o_node   = b_o_node
        self.w_ho_node  = w_ho_node

        self.delta_w_node = delta_w_node

        self.n_inputs = len(input_coords)
        self.input_coords = Tensor(input_coords, dtype=dtypes.float)

        self.n_hidden = len(hidden_coords)
        self.hidden_coords = Tensor(hidden_coords, dtype=dtypes.float)

        self.n_outputs = len(output_coords)
        self.output_coords = Tensor(output_coords, dtype=dtypes.float)

        self.weight_threshold = weight_threshold
        self.activation = activation
        self.batch_size = batch_size
        self.reset()
    def get_init_weights(self):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)
        n_in = len(in_coords)
        n_out = len(out_coords)
        zeros = Tensor.zeros((n_out, n_int), dtype=dtypes.float)
        weights = w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in, pre=zeros, post=zeros, w=zeros)
        clamp_weights(weights, self.weight_threshold)
        return weights
    def reset(self):
        Tensor.no_grad = True
        self.input_to_hidden = self.get_init_weights(self.input_coords, self.hidden_coords, self.w_ih_node)
        bias_coords = Tensor.zeros((1, 2), dtype=dtypes.float)
        self.bias_hidden = self.get_init_weights(bias_coords, self.hidden_coords, self.b_h_node).unsqueeze(0).expand(
                self.batch_size, self.n_hidden, 1
        )
        
        self.hidden_to_hidden = self.get_init_weights(self.hidden_coords, self.hidden_coords, self.w_hh_node).unsqueeze(0).expand(
                self.batch_size, self.n_hidden, self.n_hidden
        )

        bias_coords = Tensor.zeros((1, 2), dtype=dtypes.float)
        self.bias_output =  self.get_init_weights(bias_coords, self.output_coords. self.w_ho_node)

        self.hidden_to_output = self.get_init_weights(self.hidden_coords, self.output_coords, self.w_ho_node)
        self.hidden = Tensor.zeros((self.batch_size, self.n_hidden, 1), dtype=dtypes.float)

        self.batched_hidden_coords = get_coord_inputs(self.hidden_coords, self.hidden_coords, batch_size=batch_size)

    def activate(self, inputs):
        Tensor.no_grad = True
        inputs = Tensor(inputs, dtype=dtypes.float).unsqueeze(2)
        self.hidden = self.activation(self.input_to_hidden.matmul(inputs) + self.hidden_to_hidden.matmul(self.hidden) + self.bias_hidden)
        outputs = self.activation(self.hidden_to_output.matmul(self.hidden) + self.bias_output)
        hidden_outputs = self.hidden.expand(self.batch_size, self.n_hidden, self.n_hidden)
        hidden_inputs = hidden_outputs.transpose(1, 2)
        (x_out, y_out), (x_in, y_in) = self.batched_hidden_coords
        self.hidden_to_hidden += self.delta_w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in, pre=hidden_inputs, post=hidden_outputs, w=self.hidden_to_hidden)
        return outputs.squeeze(2)
    @staticmethod
    def create(genome, config, input_coords, hidden_coords, output_coords, weight_threshold=0.2, activation="tanh", batch_size=1):
        nodes = create_cppn(
                genome, config,
                ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],
                ['w_ih', 'b_h', 'w_hh', 'b_o', 'w_ho', 'delta_w']
        )
        w_ih_node = nodes[0]
        b_h_node = nodes[1]
        w_hh_node = nodes[2]
        b_o_node = nodes[3]
        w_ho_node = nodes[4]
        delta_w_node = nodes[5]

        return AdaptiveNet(w_ih_node, b_h_node, w_hh_node, b_o_node, w_ho_node, 
                           delta_w_node, input_coords, hidden_coords, output_coords, 
                           weight_threshold=weight_threshold, activation=activation, batch_size=batch_size)
