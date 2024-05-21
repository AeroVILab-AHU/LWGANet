
import os
import random
import shutil


def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def moveFile(file_dir, save_dir, rate):
    ensure_dir_exists(save_dir)
    path_dir = os.listdir(file_dir)
    filenumber = len(path_dir)
    picknumber = int(filenumber * rate)
    print(picknumber)
    sample = random.sample(path_dir, picknumber)
    # print (sample)
    for name in sample:
        shutil.move(file_dir + name, save_dir + name)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


if __name__ == '__main__':

    dataset_path = '/dataset/classification/'
    rate = 0.5
    dataset_name = 'AID'     # NWPU-RESISC45   UCM   AID


    ratename = '-' + str(int(10-rate*10)) + str(int(rate*10)) + '/'
    new_dataset_path = dataset_path + dataset_name + ratename
    dataset_dirs = os.listdir(dataset_path + dataset_name + '/')
    for file in dataset_dirs:
        file_dir = dataset_path + dataset_name + '/' + file + '/'
        print(file_dir)
        train_save_dir = new_dataset_path + '/train/' + file
        val_save_dir = new_dataset_path + '/val/' + file
        print(val_save_dir)
        val_save_dir = val_save_dir + '/'
        moveFile(file_dir, val_save_dir, rate)
        # moveFile(file_dir, train_save_dir, 1)

