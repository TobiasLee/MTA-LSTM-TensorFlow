from tensorflow.nn.rnn_cell import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.layers import core as layers_core
import tensorflow as tf


class TopicAttentionWrapper(RNNCell):
    def __init__(self, cell, memory, attention_size=128, state_is_tuple=True
                 ):
        """ Bahdanau Attention Wrapper of Topic embedding

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      memory: topic embedidng of topic words
      attention_size: size of attention
    Raises:
      TypeError: if cell is not an RNNCell.
    """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self.memory = memory  # topic word embedding
        self._state_is_tuple = state_is_tuple
        self.attention_size = attention_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):

        # Wa ht−1 +Ua topicj
        dtype = inputs.dtype
        c_t, h_t = state  # h_t batch_size x hidden_size
        embedding_size = self.memory.shape[2].value
        with vs.variable_scope("topic_attention"):
            query_layer = layers_core.Dense(self.attention_size, dtype=dtype)
            memory_layer = layers_core.Dense(self.attention_size, dtype=dtype)
            v = vs.get_variable("attention_v", [self.attention_size], dtype=dtype)
            keys = memory_layer(self.memory)  # batch_size x num x attention_size
            processed_query = array_ops.expand_dims(query_layer(h_t), 1)  # batch_size, 1 , attention_size
            score = math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])
            score = nn_ops.softmax(score, axis=1)  # softmax
            score_tile = gen_array_ops.tile(array_ops.expand_dims(score, -1), [1, 1, embedding_size],
                                            name="weight")
            mt = math_ops.reduce_sum(self.memory * score_tile, axis=1)

        return self._cell(tf.concat([inputs, mt], axis=1), state)


class MTAWrapper(RNNCell):
    def __init__(self, cell, memory, v, uf, query_layer, memory_layer, mask=None, max_len=100, attention_size=128, state_is_tuple=True
                 ):
        """ Multi-Topic aware wrapper of LSTM

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      memory: topic embedding of topic words
      mask: seq_len mask
    Raises:
      TypeError: if cell is not an RNNCell.
    """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self.memory = memory  # topic word embedding
        self._state_is_tuple = state_is_tuple
        self.attention_size = attention_size
        # tf.ones([batch_size, config.num_keywords])
        self.batch_size = self.memory.shape[0].value
        self.num_keywords = self.memory.shape[1].value
        self.embedding_size = self.memory.shape[2].value

        self.coverage_vector = array_ops.ones([self.batch_size, self.num_keywords])
        if mask is None:
            self.seq_len = array_ops.ones([self.batch_size, 1]) * max_len  # inference
        else:
            self.seq_len = math_ops.reduce_sum(mask, axis=1, keepdims=True)  # training

        self.v = v
        self.query_layer = query_layer

        self.memory_layer = memory_layer
        self.u_f = uf
        res1 = tf.sigmoid(
            tf.matmul(tf.reshape(self.memory, [self.batch_size, -1]), self.u_f))  # batch_size x num_keyword
        self.phi_res = self.seq_len * res1  # batch_size x num_keywords

        print(self.u_f)
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):

        # Wa ht−1 +Ua topicj
        c_t, h_t = state  # h_t batch_size x hidden_size
        dtype = inputs.dtype

        with vs.variable_scope("topic_attention"):

            # Attention
            keys = self.memory_layer(self.memory)  # batch_size x num x attention_size
            processed_query = array_ops.expand_dims(self.query_layer(h_t), 1)  # batch_size, 1 , attention_size
            score = self.coverage_vector * math_ops.reduce_sum(self.v * math_ops.tanh(keys + processed_query), [2])
            score = nn_ops.softmax(score, axis=1)  # softmax
            score_tile = gen_array_ops.tile(array_ops.expand_dims(score, -1), [1, 1, self.embedding_size],
                                            name="weight")
            mt = math_ops.reduce_sum(self.memory * score_tile, axis=1)

            # update coverage vector
            self.coverage_vector = self.coverage_vector - score / self.phi_res
        return self._cell(tf.concat([inputs, mt], axis=1), state)
