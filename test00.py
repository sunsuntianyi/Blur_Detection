import os

import cv2
import tensorflow as tf


class ReadImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

    @staticmethod
    def standardize_file_name(folder_path, file_format=".jpg"):
        """
        This function is used to rename all files in a folder
        """
        for i, filename in enumerate(os.listdir(folder_path)):
            os.rename(folder_path + "/" + filename, folder_path + "/" + str(i) + file_format)
        return

    @staticmethod
    def load_images_name_from_folder(folder_class_0, folder_class_1):
        """"
        This function is used to load all images of both classes
        """
        # create an empty list to store file_name
        global file_name
        folder_class_0 = folder_class_0 + '/'
        folder_class_1 = folder_class_1 + '/'
        file_name = list()
        for filename in os.listdir(folder_class_0):
            # read file base name
            # img_class_0 = ReadImage(os.path.basename(os.path.join(folder_class_0, filename)))
            # to get full path, use: (takes lots of memory)
            img_class_0 = ReadImage(os.path.join(folder_class_0, filename))
            file_name.append(str(img_class_0))

        for filename in os.listdir(folder_class_1):
            # read file base name
            # img_class_1 = ReadImage(os.path.basename(os.path.join(folder_class_1, filename)))
            # to get full path, use: (takes lots of memory)
            img_class_1 = ReadImage(os.path.join(folder_class_1, filename))
            file_name.append(str(img_class_1))
        return file_name

    @staticmethod
    def input_parser(filename, label):
        """
        This function can only be used with TensorFlow installed

        It is used for decoding (read in pixel value) and resizing image
        """
        # convert the label to one-hot encoding
        label_onehot = tf.one_hot(indices=tf.cast(label, tf.int32), depth=2)
        image_string = tf.read_file(filename)
        # decode image
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        # convert to float
        image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
        # resize image
        image_resized = tf.image.resize_images(image_decoded, [300, 300])
        return image_resized, label_onehot


# step 1
# Training data
train_folder_dir_class_0 = '/home/tianyi/Desktop/data_blur/CERTH_ImageBlurDataset/TrainingSet/undistorted'
train_folder_dir_class_1 = '/home/tianyi/Desktop/data_blur/CERTH_ImageBlurDataset/TrainingSet/blurred'

filenames_train = tf.constant(ReadImage.load_images_name_from_folder(
    folder_class_0=train_folder_dir_class_0, folder_class_1=train_folder_dir_class_1))
number_of_class_0 = len(os.listdir(train_folder_dir_class_0))
number_of_class_1 = len(os.listdir(train_folder_dir_class_1))
labels_train = tf.constant([0] * number_of_class_0 + [1] * number_of_class_1)

print(file_name)



zip the image address with their respective label


shuffle data


Validation data
val_folder_dir_class_0 = '/home/tianyi/Desktop/data_blur/CERTH_ImageBlurDataset/TrainingSet/undistorted'
val_folder_dir_class_1 = '/home/tianyi/Desktop/data_blur/CERTH_ImageBlurDataset/TrainingSet/blurred'

filenames_val = tf.constant(ReadImage.load_images_name_from_folder(
    folder_class_0=train_folder_dir_class_0, folder_class_1=train_folder_dir_class_1))
number_of_class_0 = len(os.listdir(train_folder_dir_class_0))
number_of_class_1 = len(os.listdir(train_folder_dir_class_1))
labels_val = tf.constant([0] * number_of_class_0 + [1] * number_of_class_1)

step 2: create a dataset returning slices of `filenames`
dataset_train = tf.contrib.data.Dataset.from_tensor_slices((filenames_train, labels_train))
# dataset_val = tf.contrib.data.Dataset.from_tensor_slices((filenames_val, labels_val))

# step 3: parse every image in the dataset using map
dataset = dataset_train.map(ReadImage.input_parser)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
dataset, labels = iterator.get_next()

# with tf.Session() as sess:
#     print(sess.run(dataset))
# train_init_op = iterator.make_initializer(dataset_train)
#
# with tf.Session() as sess:
#
#     # initialize the iterator on the training data
#     sess.run(train_init_op)
#
#     while True:
#         try:
#             elem = sess.run(dataset)
#             print(elem)
#         except tf.errors.OutOfRangeError:
#             print('End of training dataset')

# #
#
# def build_cnn_model(labels, mode):
#
#     # Input layer
#     input_layer = tf.reshape(images, [-1, 300, 300, 3])
#
#     # Conv. Layer 1
#     conv1 = tf.layers.conv2d(
#         name="Conv. Layer 1",
#         inputs=input_layer,
#         filters=32,
#         kernel_size=[3, 3],
#         padding="same",
#         activation=tf.nn.relu)
#
#     # Pooling Layer 1
#     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#     # Conv. Layer 2 and Pooling Layer 2
#     conv2 = tf.layers.conv2d(
#         name="Conv. Layer 2",
#         inputs=pool1,
#         filters=64,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
#
#     # Pooling Layer 2
#     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#     # Dense Layer
#     pool2_flat = tf.reshape(pool2, [-1, 126 * 126 * 64])
#     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#     dropout = tf.layers.dropout(
#         inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#     # Logit Layer
#     logit = tf.layers.dense(inputs=dropout, units=2)
#
#     predictions = {
#         # Generate predictions (for PREDICT and EVAL mode)
#         "classes": tf.argmax(input=logit, axis=1),
#         # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#         # `logging_hook`.
#         "probabilities": tf.nn.softmax(logit, name="softmax_tensor")
#     }
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     # Calculate Loss (for both TRAIN and EVAL modes)
#     onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
#     loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logit)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# with tf.Session() as sess:
#     a = sess.run([images, labels])
#     print(a)
#
