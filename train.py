import os
import sys
import cv2
import tensorflow as tf
import read_tfrecord

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training Parameters
learning_rate = 0.0001
batch_size = 32

# Network Parameters
num_input = 224
num_classes = 2
dropout = 0.5  # Dropout, probability to drop a unit


def parser(record):
    keys_to_features = {
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.float32)
    # image = tf.decode_raw(parsed["image"], tf.uint8)
    # image = tf.cast(image, tf.float32)
    # image = tf.reshape(image, shape=[224, 224, 3])
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label


def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=256)
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(1024, 1)
    )
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(parser, batch_size)
    )
    # dataset = dataset.map(parser, num_parallel_calls=12)
    # dataset = dataset.batch(batch_size=1000)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


# Create the neural network
def conv_net(features, n_classes, dropout, reuse, is_training):

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = features["image"]

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 224, 224, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.tanh)
        # Batch Normalization
        conv1 = tf.layers.batch_normalization(inputs=conv1, training=is_training)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.tanh)
        # Batch Normalization
        conv2 = tf.layers.batch_normalization(inputs=conv2, training=is_training)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.tanh)
        # Batch Normalization
        conv3 = tf.layers.batch_normalization(inputs=conv3, training=is_training)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        conv4 = tf.layers.conv2d(conv3, 256, 3, activation=tf.nn.tanh)
        # Batch Normalization
        conv4 = tf.layers.batch_normalization(inputs=conv4, training=is_training)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv4 = tf.layers.max_pooling2d(conv4, 2, 2)

        conv5 = tf.layers.conv2d(conv4, 512, 3, activation=tf.nn.tanh)
        # Batch Normalization
        conv5 = tf.layers.batch_normalization(inputs=conv5, training=is_training)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv5 = tf.layers.max_pooling2d(conv5, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.layers.flatten(conv5)
        # fc1 = tf.layers.batch_normalization(inputs=fc1, training=True)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 1024)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
        # fc1 = tf.layers.batch_normalization(inputs=fc1, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return spec


def train_input_fn():
    return input_fn(filenames=["/home/tianyi/Desktop/cat/train/training.tfrecords",
                               "/home/tianyi/Desktop/cat/validate/validation.tfrecords"])


def eval_input_fn():
    return input_fn(filenames=["/home/tianyi/Desktop/cat/test/testing.tfrecords"])


# Build the Estimator

model = tf.estimator.Estimator(model_fn=model_fn,
                               model_dir="/home/tianyi/Desktop/cat/checkpoints")


count = 0
while count < 10000:
    model.train(input_fn=train_input_fn, steps=100)
    result = model.evaluate(input_fn=eval_input_fn)
    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
    sys.stdout.flush()
    count = count + 1