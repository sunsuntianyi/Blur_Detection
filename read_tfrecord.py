import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def read_tfrecord(tfrecord_path, pixel):
    """
    This function is used to TFRecord files and extract image and its respective label information

    :param: tfrecord_path:  path of the TFRecord file
    :param: pixel:          pixel of the original image used to create this TFRecord file
                            (found in 'read_image_mat' module in 'load_images.py'
    """
    with tf.Session() as sess:
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=1)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)

        # Reshape image data into the original shape
        image = tf.reshape(image, [pixel, pixel, 3])

        # Number of record in TFRecord file
        number_of_record = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_path))

        # Creates batches by randomly shuffling tensors
        # (default: batch size = all records, no batching, but random shuffle)
        img, lbl = tf.train.shuffle_batch([image, label], batch_size=number_of_record, capacity=50000,
                                          allow_smaller_final_batch=True, num_threads=4, min_after_dequeue=10000)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        imgs, lbls = sess.run([img, lbl])
        #imgs = imgs.astype(np.uint8)

        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

    return imgs, lbls

####################################


def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label


def input_fn(filenames, train, batch_size=32, buffer_size=1024):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y


if __name__ == "__main__":

    data_path = '/home/tianyi/Desktop/skin/train/training.tfrecords'
    image_pixel = 200

    # images, labels = read_tfrecord(tfrecord_path=data_path,
    #                                pixel=image_pixel)
    #
    # print(images[1])
    # print(tf.one_hot(labels, 2))
    # # test to see if the module is correctly defined by reading an image
    # j = 200
    # plt.imshow(images[j])
    # plt.title('benign' if labels[j] == 0 else 'malignant')
    #
    #


# with tf.Session as sess:
#     sess.run(tf.global_variables_initializer())
#     data_path_train = '/home/tianyi/Desktop/skin/train/training.tfrecords'
#     x, y = input_fn(data_path_train, train=True, batch_size=32, buffer_size=1024)
#     print(x)
#     print(y)









