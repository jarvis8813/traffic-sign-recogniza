from collections import namedtuple
import tensorflow as tf
from spatial_transformer import spatial_transformer_network

Parameters = namedtuple('Parameters', [
        # Data parameters
        'num_classes', 'image_size', 
        # Training parameters
        'batch_size', 'start_epochs','max_epochs', 'log_epoch', 'print_epoch',
        # Optimisations
        'learning_rate_decay', 'learning_rate',
        'l2_reg_enabled', 'l2_lambda', 
        'early_stopping_enabled', 'early_stopping_patience', 
        'resume_training', 
        # Layers architecture
        'conv1_k', 'conv1_d', 'conv1_p', 
        'conv2_k', 'conv2_d', 'conv2_p', 
        'conv3_k', 'conv3_d', 'conv3_p', 
        'fc4_size', 'fc4_p'
    ])

def fully_connected(input, size):
    weights = tf.get_variable( 'weights', 
        shape = [input.get_shape()[1], size],
        initializer = tf.contrib.layers.xavier_initializer()
      )
    biases = tf.get_variable( 'biases',
        shape = [size],
        initializer = tf.constant_initializer(0.0)
      )
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    weights = tf.get_variable( 'weights', 
        shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
        initializer = tf.contrib.layers.xavier_initializer()
      )
    biases = tf.get_variable( 'biases',
        shape = [depth],
        initializer = tf.constant_initializer(0.0)
      )
    conv = tf.nn.conv2d(input, weights,
        strides = [1, 1, 1, 1], padding = 'SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    return tf.nn.max_pool(
        input, 
        ksize = [1, size, size, 1], 
        strides = [1, size, size, 1], 
        padding = 'SAME'
    )

def model_pass(input, params, is_training):
    """
    Performs a full model pass.

    Parameters
    ----------
    input         : Tensor
                    Batch of examples.
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    is_training   : Tensor of type tf.bool
                    Flag indicating if we are training or not (e.g. whether to use dropout).
                    
    Returns
    -------
    Tensor with predicted logits.
    """
    # in order to implement stn
    # with tf.variable_scope('stn'):
    #   with tf.variable_scope('conv1'):
    #     stnconv1 = conv_relu(input, kernel_size = 5, depth = 32)
    #     stnpool1 = pool(stnconv1, size = 2)
    #   with tf.variable_scope('conv2'):
    #     stnconv2 = conv_relu(stnpool1, kernel_size = 5, depth = 64)
    #     stnpool2 = pool(stnconv2, size = 2)
    #   with tf.variable_scope('fc1'):
    #     stnflatten = tf.reshape(stnpool2, [-1,8*8*64])
    #     fc1 = fully_connected_relu(stnflatten, size = 1024)
    #   with tf.variable_scope('theta'):
    #     theta = fully_connected(fc1, size = 6)
    #   stn_out = spatial_transformer_network(input_fmap = input, theta = theta)

    # Convolutions
    with tf.variable_scope('conv1'):
        # shape of stn is uncertain, so have to rewrite the net
        weights = tf.get_variable( 'weights', 
            shape = [params.conv1_k, params.conv1_k, 1, params.conv1_d],
            initializer = tf.contrib.layers.xavier_initializer()
          )
        biases = tf.get_variable( 'biases',
            shape = [params.conv1_d],
            initializer = tf.constant_initializer(0.0)
          )
        conv = tf.nn.conv2d(input, weights,
              strides = [1, 1, 1, 1], padding = 'SAME')
        conv1 = tf.nn.relu(conv + biases) 

        pool1 = pool(conv1, size = 2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob = params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size = params.conv2_k, depth = params.conv2_d)
        pool2 = pool(conv2, size = 2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob = params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size = params.conv3_k, depth = params.conv3_d)
        pool3 = pool(conv3, size = 2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob = params.conv3_p), lambda: pool3)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size = 4)
    #shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, 4*4*32])

    # 2nd stage output
    pool2 = pool(pool2, size = 2)
    #shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, 4*4*64])    

    # 3rd stage output
    #shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, 4*4*128])

    flattened = tf.concat([pool1, pool2, pool3],1)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size = params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob = params.fc4_p), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size = params.num_classes)
    return logits


def cal_loss(logits, labels,params):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  #labels = tf.to_int64(labels)
  if params.l2_reg_enabled:
    with tf.variable_scope('fc4', reuse = True):
        l2_loss = tf.nn.l2_loss(tf.get_variable('weights'))
  else:
    l2_loss = 0
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  loss = tf.reduce_mean(cross_entropy) + params.l2_lambda * l2_loss
  return loss

def training(loss, params):

  tf.summary.scalar('loss', loss)

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Variables that affect learning rate.
  decay_steps = 100
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(params.learning_rate,
                                  global_step,
                                  decay_steps,
                                  params.learning_rate_decay,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(lr)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  #labels = tf.to_int32(labels)
  correct = tf.nn.in_top_k(logits, labels, 1)
  correct_sum = tf.reduce_sum(tf.cast(correct, tf.int32))
  #tf.summary.scalar('correct', correct_sum)
  # Return the number of true entries.
  return correct_sum