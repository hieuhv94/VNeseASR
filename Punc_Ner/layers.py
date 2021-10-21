import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMCell

def conv1d(in_, filter_size, height, padding, is_train=True, drop_rate=0.0, scope=None):
    with tf.compat.v1.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.compat.v1.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype=tf.float32)
        bias = tf.compat.v1.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        in_ = tf.compat.v1.layers.dropout(in_, rate=drop_rate, training=is_train)
        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        xxc = tf.nn.conv2d(input=in_, filters=filter_, strides=strides, padding=padding) + bias
        out = tf.math.reduce_max(input_tensor=tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding="VALID", is_train=True, drop_rate=0.0, scope=None):
    with tf.compat.v1.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, drop_rate=drop_rate,
                         scope="conv1d_{}".format(i))
            outs.append(out)
        concat_out = tf.concat(axis=2, values=outs)
        return concat_out


class Ner_AttentionCell(RNNCell):
    def __init__(self, num_units, memory, pmemory, cell_type='n_lstm'):
        super(Ner_AttentionCell, self).__init__()
        self._cell = LSTMCell(num_units, name=cell_type)
        self.num_units = num_units
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        # Attention model
        c, m = state # c is previous cell state, m is previous hidden state
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.compat.v1.layers.dense(m, self.mem_units, use_bias=False, name="n_wah")))
        alphas = tf.squeeze(tf.exp(tf.compat.v1.layers.dense(ha, units=1, use_bias=False, name='n_way')), axis=[-1])
        alphas = tf.math.divide(alphas, tf.math.reduce_sum(input_tensor=alphas, axis=0, keepdims=True))  # (max_time, batch_size)

        w_context = tf.math.reduce_sum(input_tensor=tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        # Late fusion
        lfc = tf.compat.v1.layers.dense(w_context, self.num_units, use_bias=False, name='n_wfc')  # late fused context

        fw = tf.sigmoid(tf.compat.v1.layers.dense(lfc, self.num_units, use_bias=False, name='n_wff') +
                        tf.compat.v1.layers.dense(h, self.num_units, name='n_wfh')) # fusion weights
        
        hft = tf.math.multiply(lfc, fw) + h  # weighted fused context + hidden state
        
        return hft, new_state

class Punc_AttentionCell(RNNCell):
    def __init__(self, num_units, memory, pmemory, cell_type='p_lstm'):
        super(Punc_AttentionCell, self).__init__()
        self._cell = LSTMCell(num_units, name=cell_type)
        self.num_units = num_units
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        # Attention model
        c, m = state # c is previous cell state, m is previous hidden state
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.compat.v1.layers.dense(m, self.mem_units, use_bias=False, name="p_wah")))
        alphas = tf.squeeze(tf.exp(tf.compat.v1.layers.dense(ha, units=1, use_bias=False, name='p_way')), axis=[-1])
        alphas = tf.math.divide(alphas, tf.math.reduce_sum(input_tensor=alphas, axis=0, keepdims=True))  # (max_time, batch_size)

        w_context = tf.math.reduce_sum(input_tensor=tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        # Late fusion
        lfc = tf.compat.v1.layers.dense(w_context, self.num_units, use_bias=False, name='p_wfc')  # late fused context

        fw = tf.sigmoid(tf.compat.v1.layers.dense(lfc, self.num_units, use_bias=False, name='p_wff') +
                        tf.compat.v1.layers.dense(h, self.num_units, name='p_wfh')) # fusion weights
        
        hft = tf.math.multiply(lfc, fw) + h  # weighted fused context + hidden state
        
        return hft, new_state
