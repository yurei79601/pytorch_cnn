"""
process to generate MNIST data from numpy array to images
"""
import os
import time
import inspect
import multiprocessing as mp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from utils import check_path_exists


current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
data_path = os.path.join(current_dir, "MNIST_data")
arr_path = os.path.join(current_dir, "source_array")


def get_image_class_path(data_path, task, _class, filename):
    """get path of image"""
    return os.path.join(data_path, task, f"{int(_class)}", filename)


def save_images(data_list, task, part):
    """save images"""
    assert (
        data_list[0].shape[0] == data_list[1].shape[0]
    ), "lengths of data are not equal"

    start_number = int((data_list[0].shape[0] * (part)) / 2)
    end_number = int((data_list[0].shape[0] * (part + 1)) / 2)

    for i in range(start_number, end_number):
        image_path = get_image_class_path(
            data_path, task, data_list[1][i], f"MNIST_{i}.jpg"
        )
        cv2.imwrite(image_path, data_list[0][i])


def save_numpy_array(save_arr_path: str):
    """讀取 tensorflow 下載的檔案並且存出來"""
    check_path_exists(save_arr_path)
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    arr_list = [X_train, Y_train, X_test, Y_test]
    filename_list = ["X_train", "Y_train", "X_test", "Y_test"]
    for arr, filename in zip(arr_list, filename_list):
        np.save(os.path.join(save_arr_path, filename), arr)


def load_image_array(path: str):
    """從實體檔案讀取 numpy array of images"""
    X_train = np.load(os.path.join(path, "X_train.npy"))
    Y_train = np.load(os.path.join(path, "Y_train.npy"))
    X_test = np.load(os.path.join(path, "X_test.npy"))
    Y_test = np.load(os.path.join(path, "Y_test.npy"))
    return X_train, Y_train, X_test, Y_test


class worker(mp.Process):
    """muli-tasks worker"""

    def __init__(self, q, worker_num):
        mp.Process.__init__(self)
        self.queue = q
        self.worker_num = worker_num

    def run(self):
        while not self.queue.empty():
            data = self.queue.get()
            save_images(data[0], data[1], data[2])
            time.sleep(3)


if __name__ == "__main__":
    check_path_exists(arr_path)
    X_train, Y_train, X_test, Y_test = load_image_array(arr_path)

    task_list = ["train", "val"]
    file_type_list = ["images", "class"]

    for task in task_list:
        for file_type in file_type_list:
            check_path_exist(os.path.join(data_path, task, file_type))

    (
        features_train,
        features_test,
        targets_train,
        targets_test,
    ) = train_test_split(X_train, Y_train, test_size=0.2, random_state=123)

    train_data_list = [features_train, targets_train]
    val_data_list = [features_test, targets_test]

    queue1 = [train_data_list, "train", 0]
    queue2 = [train_data_list, "train", 1]

    queue3 = [val_data_list, "val", 0]
    queue4 = [val_data_list, "val", 1]

    my_queue = mp.Queue()

    for queue in [queue1, queue2, queue3, queue4]:
        my_queue.put(queue)

    my_worker1 = worker(my_queue, 1)
    my_worker2 = worker(my_queue, 2)
    my_worker3 = worker(my_queue, 3)
    my_worker4 = worker(my_queue, 4)

    my_worker1.start()
    my_worker2.start()
    my_worker3.start()
    my_worker4.start()

    my_worker1.join()
    my_worker2.join()
    my_worker3.join()
    my_worker4.join()
