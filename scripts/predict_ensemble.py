import argparse
import pickle
import os.path as osp
import numpy as np
from sklearn.metrics import pairwise_distances
import csv


def load_predictions(dir_path):
    test_embeddings = np.load(osp.join(dir_path, 'test_embeddings.npy'))
    gallery_embeddings = np.load(osp.join(dir_path, 'gallery_embeddings.npy'))
    with open(osp.join(dir_path, 'targets'), 'rb') as db_file:
        imgs_data = pickle.load(db_file)

    return test_embeddings, gallery_embeddings, imgs_data


def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
            '--input_dirs', type=str, nargs='+')
    parser.add_argument(
            '--method', type=str, default='concat', choices=['concat', 'distmat', 'voting'])
    parser.add_argument(
            '--metric', type=str, default='cosine')

    args = parser.parse_args()

    all_predictions = [load_predictions(p) for p in args.input_dirs]
    imgs_data = all_predictions[0][-1]
    predict_targets = imgs_data['test_targets']
    train_targets = imgs_data['gallery_targets']

    if args.method == 'distmat':
        distmat_all = 0.
        for prediction in all_predictions:
            test_embeddings, gallery_embeddings, _ = prediction
            distmat = pairwise_distances(test_embeddings, gallery_embeddings, metric=args.metric)
            distmat_all += distmat
        distmat_all /= len(all_predictions)
    elif args.method == 'concat':
        test_embeddings, gallery_embeddings, _ = all_predictions[0]
        for prediction in all_predictions[1:]:
            test, gallery, _ = prediction
            test_embeddings = np.concatenate([test_embeddings, test], axis=1)
            gallery_embeddings = np.concatenate([gallery_embeddings, gallery], axis=1)
        distmat_all = pairwise_distances(test_embeddings, gallery_embeddings, metric=args.metric)
    elif args.method == 'voting':
        pass

    distmap_sorted = np.argsort(distmat_all, axis=1).astype(int)
    distsort = np.sort(distmat_all, axis=1)[:, :5]
    t = 100.
    print(distsort.shape)
    print(np.min(distsort, axis=0))
    print(np.max(distsort, axis=0))
    print(np.mean(distsort, axis=0))
    print(np.median(distsort, axis=0))

    with open('ensemble_prediction.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(["image", "predictions"])
        for i in range(len(predict_targets)):
            candidates = []
            if distsort[i][0] > t:
                print('new', distsort[i][0])
                candidates.append('new_individual')
                candidates.extend(train_targets[distmap_sorted[i]][:4])
            else:
                candidates = train_targets[distmap_sorted[i]][:4]
                candidates = np.append(candidates, 'new_individual')

            writer.writerow([predict_targets[i], " ".join(candidates)])


if __name__ == "__main__":
    main()