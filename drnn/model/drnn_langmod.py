"""
drnn_langmod.py

Core model definition file for a Deep-Q Network Language Model with a variable action space.
The state-space input consists of a bag-of-words context vector, and the action space input consists
of a hot-encoded vector corresponding to the target word (word following the context).
"""
import tensorflow as tf


class DRNNLangmod():
    def __init__(self, vocab_size, embedding_size, hidden_size):
        """
        Instantiate the DRNNLangmod model, with the necessary parameters.

        :param vocab_size: Size of the state/action space vocabulary (for bag-of-words vectors).
        :param embedding_size: Size of intermediate state/action embeddings.
        :param hidden_size: Size of the hidden layer for the state encoding network.
        """
        self.vocab_sz, self.embedding_sz, self.hidden_sz = vocab_size, embedding_size, hidden_size

        # Setup Placeholders
        self.state_input = tf.placeholder(tf.int32, shape=[None, self.vocab_sz])  # Bag of Words
        self.action_input = tf.placeholder(tf.int32, shape=[None])                # Hot Encoding

        # Instantiate Network Parameters
        self.instantiate_weights()

        # Get state-space embeddings
        self.state_emb = self.get_state()

        # Get action embedding
        self.action_emb = self.get_action()

        # Calculate Q-Val
        self.q_val = self.get_q()

    def instantiate_weights(self):
        """
        Initialize and build all the necessary network parameters.
        """
        # Construct weights for State Encoding Network: BoW -> A -> B -> state_emb
        self.Aw = init_weight([self.vocab_sz, self.hidden_sz], "A_weight")
        self.Ab = init_bias(self.hidden_sz, 0.1, "A_bias")

        self.Bw = init_weight([self.hidden_sz, self.embedding_sz], "B_weight")
        self.Bb = init_bias(self.embedding_sz, 0.1, "B_bias")

        # Construct Embedding Matrix for Action Encoding
        self.E = init_weight([self.vocab_sz, self.embedding_sz], "E")

    def get_state(self):
        """
        Given the state-input bag of words vector, perform a forward pass through the
        state-embedding network, in order to create the state-embedding.

        :return: Returns Tensor of shape [None, embedding_size].
        """
        hidden = tf.nn.elu(tf.matmul(self.state_input, self.Aw) + self.Ab)
        return tf.nn.elu(tf.matmul(hidden, self.Bw) + self.Bb)

    def get_action(self):
        """
        Given the single word corresponding to the predicted action, look up word in embedding
        table, and return the respective embedding.

        :return: Returns Tensor of shape [None, embedding_size].
        """
        return tf.nn.embedding_lookup(self.E, self.action_input)

    def get_q(self, normalize=False):
        """
        Compute the Q-Value by taking the dot product between the state and action embeddings.

        :return: Returns Tensor of shape [None], corresponding to the Q-vals of each element in the
                 batch.
        """
        # Depending on the boolean flag, normalize the vectors using the L2 Norm
        if not normalize:
            state_norm, action_norm = self.state_emb, self.action_emb
        else:
            state_norm = tf.nn.l2_normalize(self.state_emb, dim=1)
            action_norm = tf.nn.l2_normalize(self.action_emb, dim=1)

        # Take the dot product to return the Q-values
        dot = tf.reduce_sum(state_norm * action_norm, reduction_indices=[1], name='dot')
        return dot


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