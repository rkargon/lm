"""
rl_langmod.py

Core model definition file for a Deep-Q Network Language Model. The state-space input consists of
a bag-of-words context vector, and the action space input consists of a hot-encoded vector
corresponding to the target word (word following the context).
"""
import tensorflow as tf


class RLangmod():
    def __init__(self, vocab_size):
        """
        Instantiate the RLangmod model, with the necessary parameters.

        :param vocab_size: Size of the state/action space vocabulary (for bag-of-words vectors).
        """
        self.vocab_size = vocab_size

        # Setup Placeholders
        self.state_input = tf.placeholder(tf.int32, shape=[None, self.vocab_size])  # Bag of Words
        self.action_input = tf.placeholder(tf.int32, shape=[None])                  # Hot Encoding

        # Instantiate Network Parameters
        self.instantiate_weights()

        # Get state-space embeddings
        self.state_emb = self.get_state()

        # Get action embedding
        self.action_emb = self.get_action()

    def instantiate_weights(self):
        """
        Initialize and build all the necessary network parameters.
        """
        pass

    def get_state(self):
        """
        Given the state-input bag of words vector, perform a forward pass through the
        state-embedding network, in order to create the state-embedding.

        :return: Returns Tensor of shape [None, embedding_size]
        """