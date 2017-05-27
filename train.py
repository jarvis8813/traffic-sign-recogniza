"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import tensorflow as tf
from preprocess import *
from model import *
from data_augment import *
from sklearn.cross_validation import train_test_split


#parameters

TRAIN_DATA = "./GTSRB_r" + "/Final_Training/Images"
TEST_DATA = "./GTSRB_g" + "/Final_Test/Images"
log_dir = "./log"

index_in_epoch = 0

params = Parameters(
    # Data parameters
    num_classes = 43,
    image_size = (32, 32),
    # Training parameters
    batch_size = 256,
    start_epochs = 2000,
    max_epochs = 4000,
    log_epoch = 1,
    print_epoch = 1,
    # Optimisations
    learning_rate_decay = 0.8,
    learning_rate = 0.001,
    l2_reg_enabled = True,
    l2_lambda = 0.0001,
    early_stopping_enabled = False,
    early_stopping_patience = 100,
    resume_training = True,
    # Layers architecture
    conv1_k = 5, conv1_d = 32, conv1_p = 0.9,
    conv2_k = 5, conv2_d = 64, conv2_p = 0.8,
    conv3_k = 5, conv3_d = 128, conv3_p = 0.7,
    fc4_size = 1024, fc4_p = 0.5
)

def data_nextbatch(X,y):
  batch_size = params.batch_size
  num_example = len(y)
  global index_in_epoch
  if index_in_epoch + batch_size <= num_example:
    X_batch = X[index_in_epoch:index_in_epoch+batch_size]
    y_batch = y[index_in_epoch:index_in_epoch+batch_size]
    index_in_epoch = index_in_epoch + batch_size
  elif index_in_epoch + batch_size > num_example:
    X_batch = X[index_in_epoch:]
    y_batch = y[index_in_epoch:]
    index_in_epoch = 0
  return X_batch,y_batch


def do_eval(sess,
            eval_correct,
            tf_x_batch,
            tf_y_batch,
            is_training,
            X,y):
  num_examples = len(y)
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  global index_in_epoch 
  index_in_epoch = 0   #index recount from 0
  steps_per_epoch = num_examples // params.batch_size
  #num_examples = steps_per_epoch * params.batch_size
  for step in xrange(steps_per_epoch):
    X_batch , y_batch = data_nextbatch(X,y)

    feed_dict = {
      tf_x_batch : X_batch,
      tf_y_batch : y_batch,
        is_training : False,
    }
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return precision

def run_training(restore_flag):
  X_train, y_train = load_train_data(TRAIN_DATA)
  X_test, y_test = load_test_data(TEST_DATA)
  X_train, y_train = preprocess_data(X_train, y_train)
  X_test, y_test = preprocess_data(X_test, y_test)
  #X_extend,y_extend = flip_extend(X_train, y_train)
  X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2)
  print(len(X_train))
  print(len(X_valid))
  print(len(X_test))

  # Tell TensorFlow that the model will be built into the default Graph.
  g = tf.Graph()
  with g.as_default():
    # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
    tf_x_batch = tf.placeholder(tf.float32, shape = (None, params.image_size[0], params.image_size[1], 1))
    tf_y_batch = tf.placeholder(tf.int32, shape = (None))
    is_training = tf.placeholder(tf.bool)

    # Build a Graph that computes predictions from the inference model.
    logits = model_pass(tf_x_batch,params,is_training)

    # Add to the Graph the Ops for loss calculation.
    loss = cal_loss(logits, tf_y_batch,params)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, params)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, tf_y_batch)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # And then after everything is built:
    # saver.restore(sess,log_dir+'/model.ckpt-5999')
    # print('restored!!!!!!!!!!!')
    # Run the Op to initialize the variables.
    if restore_flag == True:
      saver.restore(sess,log_dir+'/model.ckpt-1999')
      print('restored!!!!!!!!!!!')
    else:
      sess.run(init)
      print("init!!!")
      pass

    global index_in_epoch
    index_in_epoch = 0
    # Start the training loop.
    for step in xrange(params.start_epochs,params.max_epochs):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      X_batch , y_batch = data_nextbatch(X_train,y_train)
      # implement data augmentation
      #X_batch , y_batch = transform(X_batch,y_batch)


      feed_dict = {
        tf_x_batch : X_batch,
        tf_y_batch : y_batch,
          is_training : True,
      }

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 20 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 200 == 0 or (step + 1) == params.max_epochs:
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        print('Training Data Eval:')
        precision = do_eval(sess,
                eval_correct,
                tf_x_batch,
                tf_y_batch,
                is_training,
                X_train,y_train)
        tf.summary.scalar('train precision',precision)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                tf_x_batch,
                tf_y_batch,
                is_training,
                X_valid,y_valid)
        # Evaluate against the test set.
        print('Test Data Eval:')
        precision = do_eval(sess,
                eval_correct,
                tf_x_batch,
                tf_y_batch,
                is_training,
                X_test,y_test)
        tf.summary.scalar('test precision',precision)


def main(_):
  restore_flag = True
  if not restore_flag:    
    if tf.gfile.Exists(log_dir):
      tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
  run_training(restore_flag)


tf.app.run(main=main)