"""
train.py

Load the multilingual data, and build and train a multitask feed-forward language model with
the goal of predicting the next word given a fixed sequence window.

Data Credit: http://statmt.org/wmt13/translation-task.html#download
"""
from model.lm import LM
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

tf.app.flags.DEFINE_integer('vocab_size', 40000, 'Size of the vocabularies.')
tf.app.flags.DEFINE_integer('train_size', 1000000, 'Number of training examples.')
tf.app.flags.DEFINE_integer('val_size', 50000, 'Number of validation examples.')
tf.app.flags.DEFINE_integer('test_size', 100000, 'Number of test examples.')

tf.app.flags.DEFINE_integer('window_size', 5, 'Size of the fixed context window.')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Size of hidden embedding representations.')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'Size of the hidden layer.')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for Adam Optimizer.')
tf.app.flags.DEFINE_integer('batch_size', 200, 'Size of the batch size.')
tf.app.flags.DEFINE_integer('eval_every', 200, 'Print statistics every eval_every batches.')
tf.app.flags.DEFINE_float('dropout_prob', 0.5, 'Dropout keep probability.')

LANGUAGES = ['en', 'es', 'fr']
SRC_FILES = {'en': 'data/raw/english.en', 'es': 'data/raw/spanish.es', 'fr': 'data/raw/french.fr'}


def main(_):
    """
    Main training function, loads and vectorizes data, then runs the training process.
    """
    for lang_id in LANGUAGES:
        read(lang_id, SRC_FILES[lang_id], FLAGS.vocab_size, FLAGS.window_size, FLAGS.train_size,
             FLAGS.val_size, FLAGS.test_size)
    with tf.Session() as sess:
        # Instantiate Network
        print "Building Network"
        langmod = LM(LANGUAGES, FLAGS.embedding_size, FLAGS.window_size, FLAGS.hidden_size,
                     {l: FLAGS.vocab_size for l in LANGUAGES}, FLAGS.learning_rate)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        print "Initializing Variables"
        sess.run(tf.initialize_all_variables())
        bsz = FLAGS.batch_size

        print "Starting Training"

        en_x, en_y = load_train_data('en')
        fr_x, fr_y = load_train_data('fr')
        es_x, es_y = load_train_data('es')
        test_x, test_y = load_test_data('en')
        en_l, fr_l, es_l = [], [], []
        test_losses = []

        # Run 1 Epoch of French/English/Spanish With Interleaving Batches
        print 'Starting Training For Real!'
        for _ in range(50):
            start_time, loss, fr_loss, es_loss, counter = time.time(), 0.0, 0.0, 0.0, 0
            for start, end in zip(range(0, len(en_x), bsz), range(bsz, len(en_x), bsz)):
                counter += 1
                curr_french_loss, _ = sess.run(
                    [langmod.loss_vals['fr']['fr'], langmod.train_ops['fr']['fr']],
                    feed_dict={langmod.inputs['fr']: fr_x[start:end],
                               langmod.outputs['fr']: fr_y[start:end],
                               langmod.dropout_prob: FLAGS.dropout_prob})
                curr_spanish_loss, _ = sess.run(
                    [langmod.loss_vals['es']['es'], langmod.train_ops['es']['es']],
                    feed_dict={langmod.inputs['es']: es_x[start:end],
                               langmod.outputs['es']: es_y[start:end],
                               langmod.dropout_prob: FLAGS.dropout_prob})
                curr_loss, _ = sess.run(
                    [langmod.loss_vals['en']['en'], langmod.train_ops['en']['en']],
                    feed_dict={langmod.inputs['en']: en_x[start:end],
                               langmod.outputs['en']: en_y[start:end],
                               langmod.dropout_prob: FLAGS.dropout_prob})
                loss += curr_loss
                fr_loss += curr_french_loss
                es_loss += curr_spanish_loss

                # Print Evaluation Statistics
                if counter % FLAGS.eval_every == 0:
                    en_l.append(np.exp(loss / (FLAGS.eval_every)))
                    fr_l.append(np.exp(fr_loss / (FLAGS.eval_every)))
                    es_l.append(np.exp(es_loss / (FLAGS.eval_every)))
                    print '(Batch', str(counter) + ')', \
                        'English Training Perplexity:', np.exp(loss / (FLAGS.eval_every)), \
                        'Took', time.time() - start_time, 'seconds!'
                    print '(Batch', str(counter) + ')', \
                        'French Training Perplexity:', np.exp(fr_loss / (FLAGS.eval_every)), \
                        'Took', time.time() - start_time, 'seconds!'
                    print '(Batch', str(counter) + ')', \
                        'Spanish Training Perplexity:', np.exp(es_loss / (FLAGS.eval_every)), \
                        'Took', time.time() - start_time, 'seconds!'
                    start_time, loss, fr_loss, es_loss = time.time(), 0.0, 0.0, 0.0

            # Evaluate on Test Data
            test_loss, test_counter = 0.0, 0
            for start, end in zip(range(0, len(test_x), bsz), range(bsz, len(test_x), bsz)):
                test_loss += sess.run([langmod.loss_vals['en']['en']],
                                      feed_dict={langmod.inputs['en']: test_x[start:end],
                                                 langmod.outputs['en']: test_y[start:end],
                                                 langmod.dropout_prob: 1.0})[0]
                test_counter += 1
            test_loss = np.exp(test_loss / test_counter)
            test_losses.append(test_loss)
            print 'Test Perplexity:', test_loss

            # Save model
            checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, counter)

        print 'English:', en_l
        print 'French:', fr_l
        print 'Spanish:', es_l
        print 'Test:', test_losses

if __name__ == "__main__":
    tf.app.run()