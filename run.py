import create_validation_and_test_data
import load_images
import read_tfrecord

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def release_memory(a):
    del a[:]
    del a


def main(
        dataset_dir,

        train_folder_dir_class_0,
        train_folder_dir_class_1,

        validation_folder_dir_class_0,
        validation_folder_dir_class_1,

        test_folder_dir_class_0,
        test_folder_dir_class_1,

        tfrecord_dir,

        folder_name_class_0='benign',
        folder_name_class_1='malignant'
):

    tra_folder = dataset_dir + '/train'
    val_folder = dataset_dir + '/validate'
    test_folder = dataset_dir + '/test'

    # create validation class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name=folder_name_class_0)

    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name=folder_name_class_1)

    # create testing class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name=folder_name_class_0)

    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name=folder_name_class_1)

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
    images = images.astype(np.uint8)
    plt.imshow(images[j])
    plt.title('benign' if labels[j] == 0 else 'malignant')

    return


if __name__ == "__main__":

    """
    This main function is used to:
     1. Create validate and test data and their respective class sub-folder: 
            it will automatically create empty folder to store validate and test datasets
     2. Randomly pick data from train folder and move it to 
            validate and test folder with default ratio (refer to function description)
     3. Read in images from these three folders and create their TFRecord respectively
     4. Read one image from the train TFRecord file to see if the images were stored correctly 
     
    ################################### Please look at my own path below reference ###################################
    
    :param: dataset_dir:                                  Path of the dataset folder you had (downloaded)

    :param: train_folder_dir_class_0:                     Path of the train dataset for class 0 folder
    :param: train_folder_dir_class_1:                     Path of the train dataset for class 1 folder
    
    :param: validation_folder_dir_class_0:                Path of the validate dataset for class 0 folder
    :param: validation_folder_dir_class_1:                Path of the validate dataset for class 1 folder  
    
    :param: test_folder_dir_class_0:                      Path of the test dataset for class 0 folder          
    :param: test_folder_dir_class_1:                      Path of the test dataset for class 1 folder
    
    :param: tfrecord_dir:                                 TFRecord file directory for train dataset
    
    :param: folder_name_class_0: Default is 'benign', no need to change this parameter if use skin dataset
    :param: folder_name_class_1: Default is 'malignant', no need to change this parameter if use skin dataset
    """

    main(

        dataset_dir='/home/tianyi/Desktop/skin',

        train_folder_dir_class_0='/home/tianyi/Desktop/skin/train/benign',
        train_folder_dir_class_1='/home/tianyi/Desktop/skin/train/malignant',

        validation_folder_dir_class_0='/home/tianyi/Desktop/skin/validate/benign',
        validation_folder_dir_class_1='/home/tianyi/Desktop/skin/validate/malignant',

        test_folder_dir_class_0='/home/tianyi/Desktop/skin/test/benign',
        test_folder_dir_class_1='/home/tianyi/Desktop/skin/test/malignant',

        tfrecord_dir='/home/tianyi/Desktop/skin/train/training.tfrecords',
    )

