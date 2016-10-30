"""
train.py

Load the multilingual data, and build and train a multitask feed-forward language model with
the goal of predicting the next word given a fixed sequence window.

Data Credit: http://statmt.org/wmt13/translation-task.html#download
"""
from model.lm import LM, LM_RNN
from preprocessor.reader import load_test_data, load_train_data, read
import numpy as np
import os
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'data/train/', 'Path to training files.')
tf.app.flags.DEFINE_string('val_dir', 'data/val/', 'Path to validation files.')
tf.app.flags.DEFINE_string('test_dir', 'data/test/', 'Path to training files.')
tf.app.flags.DEFINE_string('vocab_dir', 'data/vocab/', 'Path to vocabularies.')
tf.app.flags.DEFINE_string('log_dir', 'data/log/', 'Path to log directory.')
tf.app.flags.DEFINE_boolean('reprocess_data', False,
                            'whether to re-process raw input data, or use pickled processed inputs. (Uses pickled '
                            'inputs by default.)')

tf.app.flags.DEFINE_integer('vocab_size', 40000, 'Size of the vocabularies.')
tf.app.flags.DEFINE_integer('train_size', 1000000, 'Number of training examples.')
tf.app.flags.DEFINE_integer('val_size', 50000, 'Number of validation examples.')
tf.app.flags.DEFINE_integer('test_size', 100000, 'Number of test examples.')

tf.app.flags.DEFINE_integer('window_size', 5, 'Size of the fixed context window.')
tf.app.flags.DEFINE_integer('rnn_steps', 20, 'Size of the fixed context window.')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Size of hidden embedding representations.')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'Size of the hidden layer.')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for Adam Optimizer.')
tf.app.flags.DEFINE_integer('batch_size', 200, 'Size of the batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs for training.')
tf.app.flags.DEFINE_integer('eval_every', 200, 'Print statistics every eval_every batches.')
tf.app.flags.DEFINE_float('dropout_prob', 0.5, 'Dropout keep probability.')

LANGUAGES = ['en', 'es', 'fr']
LANGUAGE_NAMES = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German'}
SRC_FILES = {'en': 'data/raw/english.en', 'es': 'data/raw/spanish.es', 'fr': 'data/raw/french.fr'}


def main(_):
    """
    Main training function, loads and vectorizes data, then runs the training process.
    """
    if FLAGS.reprocess_data:
        print "Processing raw data"
        for lang_id in LANGUAGES:
            read(lang_id, SRC_FILES[lang_id], FLAGS.vocab_size, FLAGS.rnn_steps, FLAGS.train_size,
                 FLAGS.val_size, FLAGS.test_size)
    with tf.Session() as sess:
        # Instantiate Network
        print "Building Network"
        langmod = LM_RNN(LANGUAGES, FLAGS.embedding_size, FLAGS.rnn_steps, FLAGS.hidden_size, FLAGS.hidden_size,
                         {l: FLAGS.vocab_size for l in LANGUAGES}, FLAGS.learning_rate)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        print "Initializing variables"
        sess.run(tf.initialize_all_variables())
        bsz = FLAGS.batch_size

        print "Loading training data..."

        # TODO test data for every language?
        train_data = {lang_id: load_train_data(lang_id) for lang_id in LANGUAGES}
        test_data = load_test_data('en')
        loss_data = {lang_id: [] for lang_id in LANGUAGES}
        test_losses = []

        # Run n epochs of French/English/Spanish With Interleaving Batches
        print 'Starting Training'
        for i in range(FLAGS.num_epochs):
            run_network(sess, langmod, train_data, loss_data, train=True, eval=True)

            # Evaluate on Test Data
            test_perplexity = run_network(sess, langmod, test_data, None, languages=["en"])
            test_losses.append(test_perplexity["en"])
            print 'Epoch %d/%d, Test Perplexity: %f' % (i, FLAGS.num_epochs, test_perplexity["en"])

            # Save model
            checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, FLAGS.train_size // bsz)

        for ld in loss_data.keys():
            print '%s: %s' % (LANGUAGE_NAMES[ld], loss_data[ld])
        print 'Test:', test_losses


def run_network(sess, langmod, data, perplexity_log=None, languages=LANGUAGES, batch_size=FLAGS.batch_size, train=False,
                eval=False):
    """
    Runs a neural network for one epoch of the given data set. This can be used either for training or for testing.
    The perplexity is also returned, and intermediate perplexity values can be stored in a given list if necessary.
    :param sess: The tensorflow session with which to run the neural net.
    :param langmod: The neural net to run
    :param data: The given data on which to run, as a dictionary of tuples {lang_id: (inputs, outputs)}
    :param perplexity_log: If not None, a dictionary mapping each language id to a list to which perplexity values
    will be appended. None by default
    :param languages: A list of language id's on which to run the network. By default uses global variable LANGUAGES
    :param batch_size: The batch size with which to pass data to the network.
    :param train: Whether or not to run a training step for each batch. False by default.
    :param eval: Whether to periodically output the perplexity every eval_size batches. If true, the returned value
    will also be the perplexity of the last eval_size batches.
    :return: The perplexity of the network on the data, as a dictionary {lang_id: perplexity}
    """
    start_time = time.time()
    counter = 0
    total_losses = {lang_id: 0.0 for lang_id in languages}
    gru_states = {lang_id: sess.run(langmod.gru_cell.zero_state(batch_size=batch_size, dtype=tf.float32)) for
                  lang_id in languages}
    for start in range(0, FLAGS.train_size, batch_size):
        end = start + batch_size
        counter += 1
        for lang_id in languages:
            inputs = data[lang_id][0][start:end]
            outputs = data[lang_id][1][start:end]
            feed_dict = {langmod.inputs[lang_id]: inputs,
                         langmod.outputs[lang_id]: outputs,
                         langmod.dropout_prob: FLAGS.dropout_prob,
                         langmod.gru_start_states[lang_id]: gru_states[lang_id]}
            batch_lang_loss = sess.run(langmod.loss_vals[lang_id][lang_id], feed_dict=feed_dict)
            if train:
                _ = sess.run(langmod.train_ops[lang_id][lang_id], feed_dict=feed_dict)
            gru_states[lang_id] = sess.run(langmod.gru_states[lang_id], feed_dict=feed_dict)
            total_losses[lang_id] += batch_lang_loss

        # Print evaluation statistics
        if eval and counter % FLAGS.eval_every == 0:
            for lang_id in languages:
                perplexity = np.exp(total_losses[lang_id] / FLAGS.eval_every)
                if perplexity_log is not None:
                    perplexity_log[lang_id].append(perplexity)
                print '(Batch %d) %s Perplexity: %f, took %f seconds!' % (
                    counter, LANGUAGE_NAMES[lang_id], perplexity, time.time() - start_time)
            total_losses = {lang_id: 0.0 for lang_id in languages}
            start_time = time.time()

    return {lang_id: np.exp(total_losses[lang_id] / counter) for lang_id in languages}


if __name__ == "__main__":
    tf.app.run()
