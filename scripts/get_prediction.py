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


def get_preprocess(img_size):
    image_size=img_size,
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
                        img_size,
                        dist_metric='cosine',
                        use_avg_embed=False,
                        rerank=False):
    test_dataset = PredictionDataset(query_img_dir, 
                                     query_imgs,
                                     query_targets,
                                     get_preprocess(img_size))
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
                                        get_preprocess(img_size))
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

def find_cutoff_thr(img_dir, imgs, targets, extractor, img_size, dist_metric='cosine'):
    dataset = PredictionDataset(img_dir, imgs, targets, get_preprocess(img_size))
    loader = torch.utils.data.DataLoader(dataset, batch_size=300, shuffle=False, num_workers=20)
    embeddings, targets = get_embeddings(loader, extractor)

    unique_targets = set(targets)

    distmat = pairwise_distances(embeddings, embeddings, metric=dist_metric)
    print(distmat.shape)
    # Holly pottato, what a wired 2d slicing to get matrixies for specific target
    other_embed = np.concatenate([distmat[np.argwhere(targets != target)].squeeze(axis=1)[:, np.argwhere(targets != target)].flatten() for target in tqdm(unique_targets)])
    target_embed = np.concatenate([distmat[np.argwhere(targets == target)].squeeze(axis=1)[:, np.argwhere(targets == target)].flatten() for target in tqdm(unique_targets)])
    print(other_embed.shape, target_embed.shape)
    print(np.min(other_embed), np.max(other_embed), np.mean(other_embed), np.median(other_embed))
    print(np.min(target_embed), np.max(target_embed), np.mean(target_embed), np.median(target_embed))
    # for target in unique_targets:
    #     other_embeddigs = distmat[np.argwhere(targets != target)].squeeze(axis=1)[:, np.argwhere(targets != target)].squeeze()
    #     target_embeddings = distmat[np.argwhere(targets == target)].squeeze(axis=1)[:, np.argwhere(targets == target)].squeeze()
    #     print("Self", target, np.mean(target_embeddings), np.median(target_embeddings), np.min(target_embeddings), np.max(target_embeddings))
    #     print("Other", target, np.mean(other_embeddigs),  np.median(other_embeddigs), np.min(other_embeddigs), np.max(other_embeddigs))


def main():
    train_img_dir = "/home/kmolchanov/reps/whales/data/images_train_cropped"
    prediction_img_dir = "/home/kmolchanov/reps/whales/data/images_test_cropped"
    seed = "3976814873"
    predict = False
    rerank = False
    test = True
    use_all = False
    use_avg_embed = False
    find_thr = False
    dist_metric='cosine'

    img_size = (128, 384)
    model_config = {
        'model_name': 'osnet_x1_0',
        'model_path': '/home/kmolchanov/reps/whales/deep-person-reid/log/custom_osnet_x1_0_dolphins_ams_256_regular_wide/model/model.pth.tar-145',
        'image_size': img_size
    }
    extractor = FeatureExtractor(**model_config, device=f'cuda')
    if find_thr:
        print("Start calculating new id thr")
        test_data = pd.read_csv(f"/home/kmolchanov/reps/whales/data/all_{seed}.csv")
        find_cutoff_thr(train_img_dir, test_data['image'].to_list(), test_data['individual_id'].to_list(), extractor, img_size, dist_metric=dist_metric)
    if test:
        test_data = pd.read_csv(f"/home/kmolchanov/reps/whales/data/test_{seed}.csv")
        gallery_data = pd.read_csv(f"/home/kmolchanov/reps/whales/data/gallery_{seed}.csv")

        # Uncomment this to get train data metrics. May be inaccurate due to cases when all samples are in test_data
        if use_all:
            all_data = pd.read_csv(f"/home/kmolchanov/reps/whales/data/all_{seed}.csv")
            test_data = all_data.sample(frac=0.2)
            gallery_data = all_data.drop(test_data.index)

        distmat, test_targets, gallery_targets = collect_precictions(train_img_dir,
                                                                     test_data['image'].to_list(),
                                                                     test_data['individual_id'].to_list(),
                                                                     train_img_dir,
                                                                     gallery_data['image'].to_list(),
                                                                     gallery_data['individual_id'].to_list(),
                                                                     extractor,
                                                                     img_size,
                                                                     dist_metric=dist_metric,
                                                                     use_avg_embed=use_avg_embed,
                                                                     rerank=rerank)

        top1 = 0
        top4 = 0
        top5 = 0
        top10 = 0
        distmap_sorted = np.argsort(distmat, axis=1).astype(int)
        distsort = np.sort(distmat, axis=1)
        query_ids = test_data['individual_id'].to_list()
        gallery_ids = set(gallery_data['individual_id'].to_list())
        count = 0
        for idx, vec in enumerate(distmap_sorted):
            # if distsort[idx][0] > 1:
            #     if query_ids[idx] not in gallery_ids:
            #         top1 += distsort[idx][0] > 0.28
            #         top4 += distsort[idx][0] > 0.28
            #         top5 += distsort[idx][0] > 0.28
            #         top10 += distsort[idx][0] > 0.28
            # else:   
            # remove examples that can't be found
            if query_ids[idx] not in gallery_ids:
                continue

            count += 1
            individs_sorted = gallery_targets[vec]
            top1 += test_targets[idx] in individs_sorted[:1]
            top4 += test_targets[idx] in individs_sorted[:4]
            top5 += test_targets[idx] in individs_sorted[:5]
            top10 += test_targets[idx] in individs_sorted[:10]
        
        print("Top1", top1 / count)
        print("Top4", top4 / count)
        print("Top5", top5 / count)
        print("Top10", top10 / count)

    if predict:
        all_data = pd.read_csv(f"/home/kmolchanov/reps/whales/data/all_{seed}.csv")
        prediction_data = pd.read_csv(f"/home/kmolchanov/reps/whales/data/test.csv")

        distmat, predict_targets, train_targets = collect_precictions(prediction_img_dir,
                                                                      prediction_data['image'].to_list(),
                                                                      prediction_data['image'].to_list(), # God of good code, please, forgive me for that
                                                                      train_img_dir,
                                                                      all_data['image'].to_list(),
                                                                      all_data['individual_id'].to_list(),
                                                                      extractor,
                                                                      img_size,
                                                                      dist_metric=dist_metric,
                                                                      use_avg_embed=use_avg_embed,
                                                                      rerank=rerank)
        
        print("Save result")
        distmap_sorted = np.argsort(distmat, axis=1).astype(int)
        distsort = np.sort(distmat, axis=1)[:, :5]
        print(distsort.shape)
        print(np.min(distsort, axis=0))
        print(np.max(distsort, axis=0))
        print(np.mean(distsort, axis=0))
        print(np.median(distsort, axis=0))

        with open('prediction.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            writer.writerow(["image","predictions"])
            for i in range(len(predict_targets)):
                candidates = train_targets[distmap_sorted[i]][:4]
                candidates = np.append(candidates, 'new_individual')
                writer.writerow([predict_targets[i], " ".join(candidates)])

if __name__ == "__main__":
    main()