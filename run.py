import create_validation_and_test_data
import load_images
import read_tfrecord
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def release_memory(a):
    del a[:]
    del a


def main(
        tra_folder='/home/tianyi/Desktop/cat/train',
        val_folder='/home/tianyi/Desktop/cat/validate',
        test_folder='/home/tianyi/Desktop/cat/test',

        train_folder_dir_class_0='/home/tianyi/Desktop/cat/train/cat',
        train_folder_dir_class_1='/home/tianyi/Desktop/cat/train/dog',

        validation_folder_dir_class_0='/home/tianyi/Desktop/cat/validate/cat',
        validation_folder_dir_class_1='/home/tianyi/Desktop/cat/validate/dog',

        test_folder_dir_class_0='/home/tianyi/Desktop/cat/test/cat',
        test_folder_dir_class_1='/home/tianyi/Desktop/cat/test/dog',

        tfrecord_dir='/home/tianyi/Desktop/cat/train/training.tfrecords'


):
    # create validation class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name='cat')

    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name='dog')

    # create testing class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name='cat')

    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name='dog')

    # split class_0 training data
    create_validation_and_test_data.split_and_move_training_data(train_dir=train_folder_dir_class_0,
                                                                 val_dir=validation_folder_dir_class_0,
                                                                 test_dir=test_folder_dir_class_0)
    # split class_1 training data
    create_validation_and_test_data.split_and_move_training_data(train_dir=train_folder_dir_class_1,
                                                                 val_dir=validation_folder_dir_class_1,
                                                                 test_dir=test_folder_dir_class_1)

    # standardize train and validation images of both classes
    load_images.standardize_file_name(cls=0,
                                      folder_dir=train_folder_dir_class_0,
                                      file_format=".jpg")
    load_images.standardize_file_name(cls=1,
                                      folder_dir=train_folder_dir_class_1,
                                      file_format=".jpg")

    load_images.standardize_file_name(cls=0,
                                      folder_dir=validation_folder_dir_class_0,
                                      file_format=".jpg")
    load_images.standardize_file_name(cls=1,
                                      folder_dir=validation_folder_dir_class_1,
                                      file_format=".jpg")

    load_images.standardize_file_name(cls=0,
                                      folder_dir=test_folder_dir_class_0,
                                      file_format=".jpg")
    load_images.standardize_file_name(cls=1,
                                      folder_dir=test_folder_dir_class_1,
                                      file_format=".jpg")

    # create TFRecord file for training set
    tra_img_path_all,\
        tra_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=train_folder_dir_class_0,
            folder_class_1=train_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=tra_folder,
                                mode='training',
                                path=tra_img_path_all,
                                labels=tra_lbls)

    # flush lists memory
    release_memory(tra_img_path_all)
    release_memory(tra_lbls)

    # create TFRecord file for validation set
    val_img_path_all,\
        val_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=validation_folder_dir_class_0,
            folder_class_1=validation_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=val_folder,
                                mode='validation',
                                path=val_img_path_all,
                                labels=val_lbls)

    # flush lists memory
    release_memory(val_img_path_all)
    release_memory(val_lbls)

    # create TFRecord file for testing set
    test_img_path_all,\
        test_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=test_folder_dir_class_0,
            folder_class_1=test_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=test_folder,
                                mode='testing',
                                path=test_img_path_all,
                                labels=test_lbls)

    # flush lists memory
    release_memory(test_img_path_all)
    release_memory(test_lbls)

    # Test if image is loaded correctly from TFRecord file
    image_pixel = 224

    images, labels = read_tfrecord.read_tfrecord(tfrecord_path=tfrecord_dir,
                                                 pixel=image_pixel)

    print(tf.one_hot(labels, 2))
    # test a random picture to see if the module is correctly defined by reading an image
    j = 1
    plt.imshow(images[j])
    plt.title('cat' if labels[j] == 0 else 'dog')

    return


if __name__ == "__main__":

    images, labels = read_tfrecord.read_tfrecord(tfrecord_path='/home/tianyi/Desktop/cat/test/testing.tfrecords',
                                                 pixel=224)

    # test a random picture to see if the module is correctly defined by reading an image
    j = 1
    # change data type from float32 to uint8 to correctly show the image color
    image = images.astype(np.uint8)
    plt.imshow(image[j])
    plt.title('cat' if labels[j] == 0 else 'dog')
