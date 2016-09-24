"""
rl_langmod.py

Core model definition file for a Deep-Q Network Language Model with a fixed action space.
The state-space input consists of a sequence of words that gets fed into an LSTM with mean-pooling,
with the goal of predicting a softmax over the action space (the entire vocabulary).
"""
import tensorflow as tf


class RLangmod():
    def __init__(self, vocab_size, embedding_size, hidden_size, max_seq_size):
        """
        Instantiate an RLangmod Deep-Q Network, with the necessary parameters.

        :param vocab_size: Size of the vocabulary (action space).
        :param embedding_size: Size of state word embeddings.
        :param hidden_size: Size of the hidden layer.
        :param max_seq_size: Maximum length of state context sequence that will get fed into model.
        """
        self.vocab_sz, self.embedding_sz = vocab_size, embedding_size
        self.hidden_sz = hidden_size
        self.max_seq_sz = max_seq_size

        # Setup Placeholders
        self.states_X = []
        for i in range(self.max_seq_sz):
            self.states_X.append(tf.placeholder(tf.int32, shape=[None], name="input{0}".format(i)))
        self.action_Y = tf.placeholder(tf.float32, shape=[None])

        # Instantiate Weights
        self.instantiate_weights()

        # Inference
        self.q_vals = self.inference()

    def instantiate_weights(self):
        """
        Instantiate Network weights and RNN Cells, for use in inference.
        """
        self.cell = tf.nn.rnn_cell.LSTMCell(self.embedding_sz)

        # Mean Pooling to Hidden
        self.h1_w = init_weight([self.embedding_sz, self.hidden_sz], 'Hidden_Weight')
        self.h1_b = init_bias(self.hidden_sz, 0.1, 'Hidden_Bias')

        # Hidden to Output
        self.o1_w = init_weight([self.hidden_sz, self.vocab_sz], 'Output_Weight')
        self.o1_b = init_bias(self.vocab_sz, 0.1, 'Output_Bias')

    def inference(self):
        """
        Build core inference computation graph, mapping the state context to the action space
        output.

        return: Final Q-Value Tensor, with values for each action
        """
        # Run the embedding cell, creating a list of length max_seq_sz of vectors of shape embed_sz
        embedding_cell = tf.nn.rnn_cell.EmbeddingWrapper(self.cell, embedding_classes=self.vocab_sz,
                                                         embedding_size=self.embedding_sz)
        outputs, _ = tf.nn.rnn(embedding_cell, self.states_X, dtype=tf.float32)

        # Mean Pooling
        mean, count = outputs[0], 0.0
        for j in range(1, len(outputs)):
            # TODO Add Padding conditional to 0-out in Mean Pooling
            mean = tf.add(mean, outputs[j])
            count += 1
        mean_pool = tf.div(mean, count)

        # Map to Hidden Layer
        hidden = tf.nn.relu(tf.matmul(mean_pool, self.h1_w) + self.h1_b)

        # Map to Output Layer
        q_vals = tf.matmul(hidden, self.o1_w) + self.o1_b
        return q_vals


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)


def init_bias(shape, value, name):
    """
    Initialize a Tensor corresponding to a bias vector with the given shape and name.

    :param shape: Shape of the bias vector (as an int, not a list).
    :param value: Value to initialize bias to.
    :param name: Name of the bias vector in the computation graph.
    :return: Tensor (Vector) object with given shape and name, initialized with given bias.
    """
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)