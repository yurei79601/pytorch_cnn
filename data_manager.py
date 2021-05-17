"""
definition of pytorch dataset for collecting classification data
e.g. MNIST
"""
import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset


class ClassficationDataset(Dataset):
    """
    Args:
        data_path: 輸入的資料路徑
        mode: 資料模式，所有的可能有 'train', 'val'
        y_label_list: 分類資料的 y 標籤清單
            e.g. 如果是 MNIST 就是
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """
    def __init__(self, data_path: str, mode: str, y_label_list: list):
        super(ClassficationDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.y_label_list = y_label_list
        self.image_list = self.get_image_list()

    def get_image_list(self) -> list:
        """回傳指定路徑的檔名"""
        image_list = []
        for y_label in self.y_label_list:
            image_list += glob(os.path.join(self.data_path, self.mode, y_label, '*.jpg'))
        return image_list

    def __len__(self) -> int:
        """回傳整個物件所收集的資料集筆數"""
        return len(self.image_list)

    def __getitem__(self, index: int):
        """
        回傳指定 index 在物件資料集的內容
        Args:
            index: 資料集的序號

        Returns:
            image: 指定資料集序號中的圖片檔 (torch.Tensor)
            y_label: 指定資料集序號中的圖片檔的分類編號 (torch.Tensor)
        """
        image = cv2.imread(self.image_list[index])
        y_label = int(self.image_list[index].split('/')[-2])
        return torch.FloatTensor(image), torch.tensor([y_label], dtype=torch.int64)


def mnist_collate(batch: int):
    """
    回傳 batch 這麼多的資料集

    Args:
        batch: 一批資料的大小，通常會是 16, 32 之類的

    Returns:
        image: 一批的圖片檔 (torch.Tensor)
            e.g. MNIST 的資料形狀會是 (batch, 28, 28, 3)
        y_label: 一批資料的 y_label (torch.Tensor)
            e.g. MNIST 的資料形狀會是 (batch, )
    """
    images, y_labels = zip(*batch)
    images = torch.stack(images, 0)
    y_labels = torch.cat(y_labels, 0)
    return images, y_labels
