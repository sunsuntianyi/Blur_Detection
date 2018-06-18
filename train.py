import os
import sys
import cv2
import tensorflow as tf


sess = tf.Session()
sess.run(tf.global_variables_initializer())


def input_parser(record):
    """
    This function can only be used with TensorFlow installed

    It is used for decoding (read in pixel value) and resizing image
    """
    keys_to_features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label':     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[300, 300, 3])
    label = tf.cast(parsed['label'], tf.int32)

    return image, label


def input_function(tfr_filename, batch_size=16, buffer_size=1024):
    global dataset
    dataset = tf.contrib.data.TFRecordDataset(filenames=tfr_filename)
    dataset = dataset.map(input_parser)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    images_batch, label_batch = iterator.get_next()

    x = {'image': images_batch}
    y = label_batch

    return x, y


tfrecord_file = '/home/tianyi/Desktop/data_blur/CERTH_ImageBlurDataset/TrainingSet/train.tfrecords'

images, labels = input_function(
    tfr_filename=tfrecord_file)


train_iterator = dataset.make_initializablle_iterator()
sess.run(train_iterator.initializer)
img, lb = sess.run([images['image'], labels])
print(img.shape, lb.shape)

