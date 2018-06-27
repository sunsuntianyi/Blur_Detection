import os
import sys
import cv2
import tensorflow as tf
import read_tfrecord


def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.

    # Reference to the tensor named "image" in the input-function.
    x = features["image"]

    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=[3, 3],
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=32, kernel_size=[3, 3],
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=32, kernel_size=[3, 3],
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)
    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                          units=num_classes)

    # Logits output of the neural network.
    logit = net

    # # Dense Layer
    # pool2_flat = tf.reshape(net, [-1, 50 * 50 * 64])
    # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(
    #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # # Logit Layer
    # logit = tf.layers.dense(inputs=dropout, units=2)

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logit)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.

        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logit)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    # predictions = {
    #     # Generate predictions (for PREDICT and EVAL mode)
    #     "classes": tf.argmax(input=logit, axis=1),
    #     # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    #     # `logging_hook`.
    #     "probabilities": tf.nn.softmax(logit, name="softmax_tensor")
    # }
    #
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    #
    # # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logit)
    #
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #     train_op = optimizer.minimize(
    #         loss=loss,
    #         global_step=tf.train.get_global_step())
    #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return spec





#
# data_path = '/home/tianyi/Desktop/skin/train/training.tfrecords'
# image_pixel = 300
#
# images, labels = read_tfrecord(tfrecord_path=data_path,
#                                pixel=image_pixel)
#
#
# labels = tf.one_hot(labels, 2)
#
# # build_cnn = build_cnn(images=images, labels=labels, mode=tf.estimator.ModeKeys.PREDICT, params=0.0001)
#
# model = tf.estimator.Estimator(model_fn=build_cnn(images=images, labels=labels, mode=tf.estimator.ModeKeys.PREDICT, params=0.0001),
#                                model_dir="/home/tianyi/Desktop/Blur_Detection/")
#
# # count = 0
# # while (count < 100000):
# #     model.train(input_fn=train_input_fn, steps=1000)
# #     result = model.evaluate(input_fn=val_input_fn)
# #     print(result)
# #     print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
# #     sys.stdout.flush()
# # count = count + 1


params = {"learning_rate": 1e-4}

img_size = 224
num_channels = 3
num_classes = 2


def train_input_fn():
    return read_tfrecord.input_fn(filenames=data_path_train, train=True)


def test_input_fn():
    return read_tfrecord.input_fn(filenames=data_path_val, train=False)


data_path_train = '/home/tianyi/Desktop/skin/train/training.tfrecords'
data_path_val = '/home/tianyi/Desktop/skin/train/training.tfrecords'

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="/home/tianyi/Desktop/skin")

model.train(input_fn=train_input_fn, steps=200)

# result = model.evaluate(input_fn=test_input_fn)
# print(result)

