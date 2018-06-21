import os
import sys
import cv2
import tensorflow as tf
from read_tfrecord import read_tfrecord


def build_cnn(images, labels, mode, params):
    num_classes = 3
    net = images

    net = tf.identity(net, name="input_tensor")

    net = tf.reshape(net, [-1, 300, 300, 3])

    net = tf.identity(net, name="input_tensor_after")

    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    net = tf.layers.dropout(net, rate=0.5, noise_shape=None,
                            seed=None, training=(mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                          units=num_classes)

    logits = net
    y_pred = tf.nn.softmax(logits=logits)

    y_pred = tf.identity(y_pred, name="output_pred")

    y_pred_cls = tf.argmax(y_pred, axis=1)

    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec


data_path = '/home/tianyi/Desktop/skin/train/training.tfrecords'
image_pixel = 300

images, labels = read_tfrecord(tfrecord_path=data_path,
                               pixel=image_pixel)


labels = tf.one_hot(labels, 2)

# build_cnn = build_cnn(images=images, labels=labels, mode=tf.estimator.ModeKeys.PREDICT, params=0.0001)

model = tf.estimator.Estimator(model_fn=build_cnn(images=images, labels=labels, mode=tf.estimator.ModeKeys.PREDICT, params=0.0001),
                               model_dir="/home/tianyi/Desktop/Blur_Detection/")

# count = 0
# while (count < 100000):
#     model.train(input_fn=train_input_fn, steps=1000)
#     result = model.evaluate(input_fn=val_input_fn)
#     print(result)
#     print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
#     sys.stdout.flush()
# count = count + 1

