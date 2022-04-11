import multiprocessing
import os
import os.path as osp
import glob
import csv
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import pairwise_distances
from torchreid.utils import FeatureExtractor
from torch import multiprocessing
from collections import defaultdict
from PIL import Image
from joblib import Parallel, delayed

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, read_image, re_ranking
)

class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, images, targets, transforms) -> None:
        super().__init__()

        self.img_root = img_root
        self.image_list = images
        self.targets = targets

        self.t = transforms
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.img_root, self.image_list[idx]))
        if self.t:
            img = self.t(img)
        
        target = None
        if self.targets:
            target = self.targets[idx]
        
        return img, target


def get_preprocess():
    image_size=(256, 512),
    pixel_mean=[0.485, 0.456, 0.406]
    pixel_std=[0.229, 0.224, 0.225]
    pixel_norm=True
    transforms = []
    transforms += [T.Resize(image_size)]
    transforms += [T.ToTensor()]
    if pixel_norm:
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    return T.Compose(transforms)

def get_embeddings(dataloader, extractor):
    embeddings = []
    targets = []
    for batch in tqdm(dataloader):
        imgs, inds = batch
        embeddings.extend(extractor(imgs).cpu().numpy())
        targets.extend(inds)
    embeddings = np.array(embeddings)
    targets = np.array(targets)
    return embeddings, targets

def collect_precictions(query_img_dir, 
                        query_imgs: list, 
                        query_targets: list, 
                        gallery_img_dir,
                        gallery_imgs,
                        gallery_targets,
                        extractor,
                        dist_metric='cosine',
                        use_avg_embed=False,
                        rerank=False):
    test_dataset = PredictionDataset(query_img_dir, 
                                     query_imgs,
                                     query_targets,
                                     get_preprocess())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=300, 
                                            shuffle=False,
                                            num_workers=30)

    test_embeddings, test_targets = get_embeddings(test_loader, extractor)
    print("Test mbeddings shape", test_embeddings.shape)
    print("Test targets shape", test_targets.shape)


    gallery_dataset = PredictionDataset(gallery_img_dir, 
                                        gallery_imgs,
                                        gallery_targets,
                                        get_preprocess())
    gallery_loader = torch.utils.data.DataLoader(gallery_dataset,
                                                batch_size=300, 
                                                shuffle=False,
                                                num_workers=30)


    gallery_embeddings, gallery_targets = get_embeddings(gallery_loader, extractor)
    print("Gallery mbeddings shape", gallery_embeddings.shape)
    print("Gallery targets shape", gallery_targets.shape)

    if use_avg_embed:
        new_embeddings = []
        new_targets = []
        for id in np.unique(gallery_targets):
            embedings = gallery_embeddings[gallery_targets==id]
            new_embedding = np.mean(embedings, axis=0)
            new_embeddings.append(new_embedding)
            new_targets.append(id)
        gallery_embeddings = np.array(new_embeddings)
        gallery_targets = np.array(new_targets)
        
        print("Gallery beddings shape", gallery_embeddings.shape)
        print("Gallery targets shape", gallery_targets.shape)

    distmat = pairwise_distances(test_embeddings, gallery_embeddings, metric=dist_metric)
    if rerank:
        print("Re ranking")
        distmat_qq = metrics.compute_distance_matrix(torch.Tensor(test_embeddings), torch.Tensor(test_embeddings), dist_metric)
        distmat_gg = metrics.compute_distance_matrix(torch.Tensor(gallery_embeddings), torch.Tensor(gallery_embeddings), dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    
    return distmat, test_targets, gallery_targets

def main():
    train_img_dir = "/home/kmolchanov/reps/whales/data/images_train_cropped"
    prediction_img_dir = "/home/kmolchanov/reps/whales/data/images_test_cropped"
    test = True
    predict = True
    rerank = False
    use_avg_embed = True
    dist_metric='cosine'

    model_config = {
        'model_name': 'osnet_x1_0',
        'model_path': '/home/kmolchanov/reps/whales/runs/logs/reid/custom_osnet_x1_0_dolphins_softmax_cosinelr_512_body_improved/model/model.pth.tar-145',
        'image_size': (256, 512)
    }
    extractor = FeatureExtractor(**model_config, device=f'cuda')

    if test:
        test_data = pd.read_csv("/home/kmolchanov/reps/whales/data/test_right_1.csv")
        gallery_data = pd.read_csv("/home/kmolchanov/reps/whales/data/gallery_right_1.csv")
        all_data = pd.read_csv("/home/kmolchanov/reps/whales/data/all_right_1.csv")

        # Uncomment this to get train data metrics. May be inaccurate due to cases when all samples are in test_data
        test_data = all_data.sample(frac=0.2)
        gallery_data = all_data.drop(test_data.index)
        distmat, test_targets, gallery_targets = collect_precictions(train_img_dir,
                                                                     test_data['image'].to_list(),
                                                                     test_data['individual_id'].to_list(),
                                                                     train_img_dir,
                                                                     gallery_data['image'].to_list(),
                                                                     gallery_data['individual_id'].to_list(),
                                                                     extractor,
                                                                     dist_metric=dist_metric,
                                                                     use_avg_embed=use_avg_embed,
                                                                     rerank=rerank)

        top1 = 0
        top4 = 0
        top5 = 0
        top10 = 0
        distmap_sorted = np.argsort(distmat, axis=1).astype(int)
        for idx, vec in enumerate(distmap_sorted):
            individs_sorted = gallery_targets[vec]
            top1 += test_targets[idx] in individs_sorted[:1]
            top4 += test_targets[idx] in individs_sorted[:4]
            top5 += test_targets[idx] in individs_sorted[:5]
            top10 += test_targets[idx] in individs_sorted[:10]
        
        print("Top1", top1 / len(test_targets))
        print("Top4", top4 / len(test_targets))
        print("Top5", top5 / len(test_targets))
        print("Top10", top10 / len(test_targets))

    if predict:
        all_data = pd.read_csv("/home/kmolchanov/reps/whales/data/all_right_1.csv")
        prediction_data = pd.read_csv("/home/kmolchanov/reps/whales/data/test.csv")

        distmat, predict_targets, train_targets = collect_precictions(prediction_img_dir,
                                                                      prediction_data['image'].to_list(),
                                                                      prediction_data['image'].to_list(), # God of good code, please, forgive me for that
                                                                      train_img_dir,
                                                                      all_data['image'].to_list(),
                                                                      all_data['individual_id'].to_list(),
                                                                      extractor,
                                                                      dist_metric=dist_metric,
                                                                      use_avg_embed=use_avg_embed,
                                                                      rerank=rerank)
        
        print("Save result")
        distmap_sorted = np.argsort(distmat, axis=1).astype(int)
        with open('prediction_new.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            writer.writerow(["image","predictions"])
            for i in range(len(predict_targets)):
                candidates = train_targets[distmap_sorted[i]][:4]
                candidates = np.append(candidates, 'new_individual')
                writer.writerow([predict_targets[i], " ".join(candidates)])

if __name__ == "__main__":
    main()