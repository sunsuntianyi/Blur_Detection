import create_validation_and_test_data
import load_images
import read_tfrecord
import tensorflow as tf
import matplotlib.pyplot as plt


def main(
        tra_folder='/home/tianyi/Desktop/skin/train',
        val_folder='/home/tianyi/Desktop/skin/validate',
        test_folder='/home/tianyi/Desktop/skin/test',

        train_folder_dir_class_0='/home/tianyi/Desktop/skin/train/benign',
        train_folder_dir_class_1='/home/tianyi/Desktop/skin/train/malignant',

        validation_folder_dir_class_0='/home/tianyi/Desktop/skin/validate/benign',
        validation_folder_dir_class_1='/home/tianyi/Desktop/skin/validate/malignant',

        test_folder_dir_class_0='/home/tianyi/Desktop/skin/validate/benign',
        test_folder_dir_class_1='/home/tianyi/Desktop/skin/validate/malignant',

        tfrecord_dir='/home/tianyi/Desktop/skin/train/training.tfrecords'

):
    # create validation class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name='benign')

    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name='malignant')

    # create testing class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name='benign')

    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name='malignant')

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
    tra_img_name_all,\
        tra_img_mat_all,\
        tra_img_path_all,\
        tra_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=train_folder_dir_class_0,
            folder_class_1=train_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=tra_folder,
                                mode='training',
                                path=tra_img_path_all,
                                labels=tra_lbls)

    # create TFRecord file for validation set
    val_img_name_all,\
        val_img_mat_all,\
        val_img_path_all,\
        val_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=validation_folder_dir_class_0,
            folder_class_1=validation_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=val_folder,
                                mode='validation',
                                path=val_img_path_all,
                                labels=val_lbls)

    # create TFRecord file for testing set
    test_img_name_all,\
        test_img_mat_all,\
        test_img_path_all,\
        test_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=test_folder_dir_class_0,
            folder_class_1=test_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=test_folder,
                                mode='testing',
                                path=test_img_path_all,
                                labels=test_lbls)

    # Test if image is loaded correctly from TFRecord file
    image_pixel = 224

    images, labels = read_tfrecord.read_tfrecord(tfrecord_path=tfrecord_dir,
                                                 pixel=image_pixel)

    print(tf.one_hot(labels, 2))
    # test to see if the module is correctly defined by reading an image
    j = 200
    plt.imshow(images[j])
    plt.title('benign' if labels[j] == 0 else 'malignant')


