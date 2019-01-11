from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.nn.rnn_cell import RNNCell
import tensorflow as tf


class SC_DropoutWrapper(DropoutWrapper):
    def __init__(self, cell, *args, **kwargs):
        DropoutWrapper.__init__(self, cell, *args, **kwargs)
        if not isinstance(cell, SCLSTM):
            raise TypeError("The wrapper is only designed for SCLSTM.")
        # self._cell = cell

    def __call__(self, inputs, state, d_act):
        if not self._cell.built:
            self._cell.build(inputs.shape)
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                self._input_keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state, d_act)
        if (not isinstance(self._output_keep_prob, float) or
                self._output_keep_prob < 1):
            output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state


class ActionWrapper(RNNCell):
    def __init__(self, cell, action_vec, wr, hr):
        if not isinstance(cell, SC_DropoutWrapper) and not isinstance(cell, SCLSTM):
            raise TypeError("The wrapper is only designed for SCLSTM.")
        self._cell = cell
        self.action_vec = action_vec  # initial one-hot action vector
        self.wr = wr  # [word_embedding_size, topic_size ]
        self.hr = hr  # [hidden_size, topic_size]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if not self._cell.built:
            print("building... cell")
            self._cell.build(inputs.shape)

        ct, ht = state
        # compute sigmoid and update action vec
        # r_t = sigmoid( W_wr x_t + W_hr h_{t-1})  = sigmoid(e1 + e2)
        e1 = math_ops.matmul(inputs, self.wr)  # [batch_size, topic_size]
        e2 = math_ops.matmul(ht, self.hr)
        r_t = math_ops.sigmoid(math_ops.add(e1, e2))  # [batch_size, topic_size]
        # update action vector
        self.action_vec = r_t * self.action_vec
        return self._cell(inputs, state, self.action_vec)


class SCLSTM(BasicLSTMCell):
    def __init__(self, kwd_voc_size, *args, **kwargs):
        BasicLSTMCell.__init__(self, *args, **kwargs)
        self.key_words_voc_size = kwd_voc_size
        print("initialized")

    def __call__(self, inputs, state, d_act):
        """
        :param inputs:
        :param state:
        :param d_act: one-hot action vector
        :return:
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)

        w_d = vs.get_variable('w_d', [self.key_words_voc_size, self._num_units])
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        #  take action vector into account
        new_c = add(add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                        multiply(sigmoid(i), self._activation(j))),
                    math_ops.tanh(math_ops.matmul(d_act, w_d)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state
