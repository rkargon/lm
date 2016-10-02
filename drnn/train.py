"""
train.py

Core training file for the Deep-Q Network Language Model with variable action space (DRNN).
"""
from model.drnn_langmod import DRNNLangmod
import numpy as np
import random
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_integer('vocabulary_size', 10000, "Size of the language vocabulary.")
tf.flags.DEFINE_integer('embedding_size', 100, "Size of intermediate embeddings.")
tf.flags.DEFINE_integer('hidden_size', 1000, "Size of state encoding network hidden layer.")

tf.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to run through training data.')
tf.flags.DEFINE_float('epsilon', 1.0, 'Starting epsilon value for random actions.')

START_TOKEN = ">>>"


def main(_):
    """
    Main function, loads and vectorizes the data, instantiates the network variables, and runs the
    training process.
    """
    # Perform data preprocessing here
    # TODO

    with tf.Session() as sess:
        # Instantiate Network
        print 'Building Network!'
        drnn = DRNNLangmod(FLAGS.vocabulary_size, FLAGS.embedding_size, FLAGS.hidden_size)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        print "Initializing Variables"
        sess.run(tf.initialize_all_variables())

        # Pseudocode for training
        sentences = []
        for _ in range(FLAGS.num_epochs):
            for sentence in sentences:
                actual_state, network_state = [], []
                for i in range(len(sentence) + 1):
                    if i == 0:
                        actual_state.append(START_TOKEN)
                        network_state.append(START_TOKEN)
                    else:
                        actual_state.append(sentence[i])

                    # TODO: Epsilon Coin Flip
                    if random.random() < FLAGS.epsilon:
                        action = np.random.choice(FLAGS.vocabulary_size)
                    else:
                        q_vals = []
                        for act in range(FLAGS.vocabulary_size):
                            q_vals.append(sess.run(drnn.q_val, feed_dict={
                                drnn.state_input: network_state, drnn.action_input: act}))
                        action = np.argmax(q_vals)
                    network_state.append(action)                 # TODO: Index in Vocab

                    # TODO: Negative Cross-Entropy Loss (Scale Bag of Words) for Reward



if __name__ == "__main__":
    tf.app.run()