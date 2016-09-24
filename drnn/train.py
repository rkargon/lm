"""
train.py

Core training file for the Deep-Q Network Language Model.
"""
from model.rl_langmod import RLangmod
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_integer('vocabulary_size', 10000, "Size of the language vocabulary.")
tf.flags.DEFINE_integer('embedding_size', 100, "Size of intermediate embeddings.")
tf.flags.DEFINE_integer('hidden_size', 1000, "Size of state encoding network hidden layer.")


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
        rlangmod = RLangmod(FLAGS.vocabulary_size, FLAGS.embedding_size, FLAGS.hidden_size)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        print "Initializing Variables"
        sess.run(tf.initialize_all_variables())


if __name__ == "__main__":
    tf.app.run(main())