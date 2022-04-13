from __future__ import division, print_function, absolute_import
import enum
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

        seed = '3976814873'
        self.train_ann_file = osp.join(self.dataset_dir, f"train_{seed}.csv")
        self.test_ann_file = osp.join(self.dataset_dir, f"test_{seed}.csv")
        self.gallery_ann_file = osp.join(self.dataset_dir, f"gallery_{seed}.csv")

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
        hist_data = {}
        with open(ann_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                img_path = os.path.join(dir_path, row['image'])
                data.append((img_path, int(row['id']), int(row['camera_id']), 0, int(row['species_id'])))

            if mode == "train":
                for i, item in enumerate(data):
                    if item[1] in hist_data:
                        hist_data[item[1]] += 1
                    else:
                        hist_data[item[1]] = 1

                if False:
                    print(len(species_map))
                    import matplotlib.pyplot as plt
                    hist_data = list(sorted(hist_data.values()))
                    print(hist_data)
                    plt.hist(hist_data, bins=100)
                    plt.show()
                    print(len(list(filter(lambda x: x > 4, hist_data))))
                    exit(0)


        return data
