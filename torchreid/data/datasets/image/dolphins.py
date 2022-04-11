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

        self.train_ann_file = osp.join(self.dataset_dir, "train_right_1.csv")
        self.test_ann_file = osp.join(self.dataset_dir, "test_right_1.csv")
        self.gallery_ann_file = osp.join(self.dataset_dir, "gallery_right_1.csv")

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
        ann_file = None
        if mode == "train":
            ann_file = self.train_ann_file
        elif mode == "test":
            ann_file = self.test_ann_file
        elif mode == "gallery":
            ann_file = self.gallery_ann_file

        data = []
        missing_count = 0
        with open(ann_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            id_to_idx = {}
            last_id = 0
            for row in reader:
                img_path = os.path.join(dir_path, row['image'])
                if not osp.isfile(img_path):
                    missing_count += 1
                    continue
                raw_id = int(float(row['id']))
                if raw_id not in id_to_idx:
                    id_to_idx[raw_id] = last_id
                    last_id += 1
                if mode == "train": id = id_to_idx[raw_id]
                else: id = raw_id
                data.append((img_path, id, int(float(row['camera_id']))))
        print(f'Missing items: {missing_count}')

        return data
