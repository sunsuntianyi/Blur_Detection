import os
import sys
import cv2
import tensorflow as tf
import numpy as np


def read_image_mat(path, resize=300):
    """
    This function is used to read image into matrix

    :param: path:     path of the image
    :param: resize:   resize image pixel, default is 300
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def standardize_file_name(cls, folder_dir, file_format='.jpg'):
    """
    This function is used to rename all files in a folder

    :param: cls:          class of the images
    :param: folder_dir:   folder directory of that class images
    :param: file_format:  image format, default: '.jpg'
    """
    if cls == 0:
        for i, filename in enumerate(sorted(os.listdir(folder_dir), key=lambda name: int(name[8:-4]))):
            os.rename(os.path.join(folder_dir, filename), folder_dir + "/" + "class_0_" + str(i) + file_format)
        return

    if cls == 1:
        for i, filename in enumerate(sorted(os.listdir(folder_dir), key=lambda name: int(name[8:-4]))):
            os.rename(os.path.join(folder_dir, filename), folder_dir + "/" + "class_1_" + str(i) + file_format)
        return
    return


def load_images_and_label_from_folder(folder_class_0, folder_class_1):
    """"
    This function is used to load all images name, image path and labels of both classes

    :param: folder_class_0: folder path of class_0 images
    :param: folder_class_1: folder path of class_1 images
    """
    # create an empty list to store file_name
    folder_class_0 = folder_class_0 + '/'
    folder_class_1 = folder_class_1 + '/'
    # file_name_0, file_name_1, file_name = list()
    img_mat_class_0, img_mat_class_1, img_mat_all = [], [], []
    img_name_class_0, img_name_class_1, img_name_all = [], [], []
    image_path_class_0, image_path_class_1, image_path_all = [], [], []

    for filename in os.listdir(folder_class_0):
        # read image name for class 0
        read_img_name_class_0 = os.path.basename(os.path.join(folder_class_0, filename))
        img_name_class_0.append(read_img_name_class_0)
        # read image matrix for class 0
        read_img_mat_class_0 = read_image_mat(os.path.join(folder_class_0, filename))
        img_mat_class_0.append(read_img_mat_class_0)
        # read image path for class 0
        read_img_path_class_0 = os.path.abspath(os.path.join(folder_class_0, filename))
        image_path_class_0.append(read_img_path_class_0)

    # sort the list based on the in-between number 1,2,3 etc.. eg. class_0_1.jpg, class_0_2.jpg
    img_name_class_0 = sorted(img_name_class_0, key=lambda name: int(name[8:-4]))

    for filename in os.listdir(folder_class_1):
        # read image name for class 1
        read_img_name_class_1 = os.path.basename(os.path.join(folder_class_1, filename))
        img_name_class_1.append(read_img_name_class_1)
        # read image matrix for class 1
        read_img_mat_class_1 = read_image_mat(os.path.join(folder_class_1, filename))
        img_mat_class_1.append(read_img_mat_class_1)
        # read image path for class 1
        read_img_path_class_1 = os.path.abspath(os.path.join(folder_class_1, filename))
        image_path_class_1.append(read_img_path_class_1)

    # sort the list based on the in-between number 1,2,3 etc.. eg. class_1_1.jpg, class_1_2.jpg
    img_name_class_1 = sorted(img_name_class_1, key=lambda name: int(name[8:-4]))

    # # combine the two lists to create the final lists for both classes
    img_mat_all = img_mat_class_0 + img_mat_class_1
    img_name_all = img_name_class_0 + img_name_class_1
    img_path_all = image_path_class_0 + image_path_class_1

    # create and load label
    number_of_class_0 = len(os.listdir(folder_class_0))
    number_of_class_1 = len(os.listdir(folder_class_1))
    lbls = [0] * number_of_class_0 + [1] * number_of_class_1

    return img_name_all, img_mat_all, img_path_all, lbls


def create_tfrecord(folder_path, mode, path, labels):
    """
    This function is used to create TFRecord file, which is a binary file,
    to store all the images of both classes and its' respective label.

    This function can only be used with TensorFlow installed

    :param: folder_path: folder path of the TFRecord file you want to save to
    :param: mode:        create a train TFRecord file or validate TFRecord file. options: 'training' or 'validation'
    :param: path:        path of the returned value in 'load_images_and_label_from_folder' module, eg. img_path_all
    :param: labels:      label of the returned value in 'load_images_and_label_from_folder' module, eg. lbls
    """
    tfwriter = tf.python_io.TFRecordWriter(folder_path + '/' + mode + '.tfrecords')
    for i in range(len(path)):
        # print for every 10 loops
        if not i % 10:
            print('Read ' + mode + ' data {}/{}'.format(i, len(path)))
            sys.stdout.flush()

        # feed in images
        img = read_image_mat(path[i])
        label = labels[i]

        # skip to next step if no more images to read
        if img is None:
            continue

        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        # create a protocol buffer
        protocol = tf.train.Example(features=tf.train.Features(feature=feature))
        # serialize to string and write to the file
        tfwriter.write(protocol.SerializeToString())

    tfwriter.close()
    sys.stdout.flush()

    return


if __name__ == "__main__":

    train_folder_dir_class_0 = '/home/tianyi/Desktop/skin/train/benign'
    train_folder_dir_class_1 = '/home/tianyi/Desktop/skin/train/malignant'

    validation_folder_dir_class_0 = '/home/tianyi/Desktop/skin/validate/benign'
    validation_folder_dir_class_1 = '/home/tianyi/Desktop/skin/validate/malignant'

    # The below four lines of code are one-time use to standardize the raw image file names
    # in class 0 and class 1 folder of the training and validation dataset (you can comment out after the first run)
    standardize_file_name(cls=0,
                          folder_dir=train_folder_dir_class_0,
                          file_format=".jpg")
    standardize_file_name(cls=1,
                          folder_dir=train_folder_dir_class_1,
                          file_format=".jpg")

    standardize_file_name(cls=0,
                          folder_dir=validation_folder_dir_class_0,
                          file_format=".jpg")
    standardize_file_name(cls=1,
                          folder_dir=validation_folder_dir_class_1,
                          file_format=".jpg")

    # create TFRecord file for training set
    tra_img_name_all, \
        tra_img_mat_all, \
        tra_img_path_all, \
        tra_lbls = \
        load_images_and_label_from_folder(
            folder_class_0=train_folder_dir_class_0,
            folder_class_1=train_folder_dir_class_1)

    tra_folder = '/home/tianyi/Desktop/skin/train'
    create_tfrecord(folder_path=tra_folder,
                    mode='training',
                    path=tra_img_path_all,
                    labels=tra_lbls)

    # create TFRecord file for validation set
    val_img_name_all, \
        val_img_mat_all, \
        val_img_path_all, \
        val_lbls = \
        load_images_and_label_from_folder(
            folder_class_0=validation_folder_dir_class_0,
            folder_class_1=validation_folder_dir_class_1)

    val_folder = '/home/tianyi/Desktop/skin/validate'
    create_tfrecord(folder_path=val_folder,
                    mode='validation',
                    path=val_img_path_all,
                    labels=val_lbls)

    # # test to see if the module is correctly defined by reading an image
    # j = 304
    # print(val_img_mat_all[j])
    # print(val_img_name_all[j])
    # print(val_lbls[j])

