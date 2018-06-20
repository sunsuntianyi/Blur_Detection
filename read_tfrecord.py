import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = '/home/tianyi/Desktop/data_blur/CERTH_ImageBlurDataset/TrainingSet/train.tfrecords'
image_pixel = 224


def read_tfrecord(tfrecord_path):
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
        image = tf.reshape(image, [image_pixel, image_pixel, 3])

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
        imgs = imgs.astype(np.uint8)

        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

    return imgs, lbls


images, labels = read_tfrecord(tfrecord_path=data_path)

j = 200
plt.imshow(images[j])
plt.title('undistorted' if labels[j] == 0 else 'blurred')















