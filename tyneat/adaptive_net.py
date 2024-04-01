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
    def reset(self):pass
