import os
import numpy as np
import shutil
from load_images import standardize_file_name
from natsort import natsorted # use this for hard sorting


def create_folder(data_dir, folder_name):
    """
    This function is used to create folder

    :param: data_dir:     path where you want to put your folder
    :param: folder_name:  name of the folder you want to create
    """
    # create folder if the folder does not exist
    if not os.path.exists(data_dir + '/' + folder_name):
        os.makedirs(data_dir + '/' + folder_name)
    else:
        print("folder already exists")
    return


def split_and_move_training_data(train_dir, val_dir, portion=0.4):
    """
    This function is used to split data from the training set to validation set
    (if you do not have validation set)

    :param: train_dir:  training data folder directory
    :param: val_dir:    validation data folder directory
    :param: portion:    proportion of the training data which you want to move to validation folder, default: 40%
    """
    all_file_path = []
    # read all file names in the train_dir
    for filename in os.listdir(train_dir):
        read_list = os.path.abspath(os.path.join(train_dir, filename))
        all_file_path.append(read_list)

    # random select files
    val_list = np.random.choice(all_file_path, replace=False, size=int(np.floor(portion * len(all_file_path))))

    # move the randomly selected file from train_dir to val_dir
    for i in range(len(val_list)):
        # break to prevent repetitive user run in future
        if len(os.listdir(val_dir)) <= portion * len(os.listdir(train_dir)):
            shutil.move(val_list[i], val_dir + '/')
        else:
            print('Validation file already moved, please check')
            break
    return


if __name__ == "__main__":

    # create validation class_0 and class_1 sub-folder
    val_path = '/home/tianyi/Desktop/skin/validate'

    create_folder(data_dir=val_path,
                  folder_name='benign')
    create_folder(data_dir=val_path,
                  folder_name='malignant')

    # folder path
    train_folder_dir_class_0 = '/home/tianyi/Desktop/skin/train/benign'
    validation_folder_dir_class_0 = '/home/tianyi/Desktop/skin/validate/benign'

    train_folder_dir_class_1 = '/home/tianyi/Desktop/skin/train/malignant'
    validation_folder_dir_class_1 = '/home/tianyi/Desktop/skin/validate/malignant'

    # standardize file name is both training class_0 and 1 folder
    standardize_file_name(cls=0,
                          folder_dir=train_folder_dir_class_0,
                          file_format=".jpg")
    standardize_file_name(cls=1,
                          folder_dir=train_folder_dir_class_1,
                          file_format=".jpg")

    # split class_0 training data
    split_and_move_training_data(train_dir=train_folder_dir_class_0,
                                 val_dir=validation_folder_dir_class_0,
                                 portion=0.4)
    # split class_1 training data
    split_and_move_training_data(train_dir=train_folder_dir_class_1,
                                 val_dir=validation_folder_dir_class_1,
                                 portion=0.4)
