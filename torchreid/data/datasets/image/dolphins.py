from __future__ import division, print_function, absolute_import
import os
import glob
import os.path as osp
import csv

from ..dataset import ImageDataset


class Dolphins(ImageDataset):
    """Dolphins.

    """

    train_dir = "images_train_cropped"
    test_dir = "images_train_cropped"

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = root
        self.train_dir = osp.join(
            self.dataset_dir, self.train_dir
        )
        self.query_dir = osp.join(
            self.dataset_dir, self.test_dir
        )
        self.gallery_dir = osp.join(
            self.dataset_dir, self.train_dir
        )

        self.train_ann_file = osp.join(self.dataset_dir, "train_right.csv")
        self.test_ann_file = osp.join(self.dataset_dir, "test_right.csv")
        self.gallery_ann_file = osp.join(self.dataset_dir, "gallery_right.csv")

        required_files = [
            self.dataset_dir, 
            self.train_dir, 
            self.query_dir, 
            self.gallery_dir, 
            self.train_ann_file,
            self.test_ann_file
        ]
        self.check_before_run(required_files)

        self.fake_camid = 0
        self.img_to_id = {}
        self.id_to_pid = {}
        self.pid_counter = 0
        train = self.process_dir(self.train_dir, mode="train")
        query = self.process_dir(self.query_dir, mode="test")
        gallery = self.process_dir(self.gallery_dir, mode="gallery")
        super(Dolphins, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, mode):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        ann_file = None
        if mode == "train":
            ann_file = self.train_ann_file
        elif mode == "test":
            ann_file = self.test_ann_file
        elif mode == "gallery":
            ann_file = self.gallery_ann_file

        data = []
        with open(ann_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                img_path = os.path.join(dir_path, row['image'])
                data.append((img_path, int(float(row['id'])), int(float(row['camera_id']))))

        return data
