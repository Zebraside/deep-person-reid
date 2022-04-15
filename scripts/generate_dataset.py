import pandas as pd
import numpy as np
import random
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/kmolchanov/reps/whales/data"
TRAIN_FILE_PATH = os.path.join(ROOT, "train.csv")
TRAIN_IMG_PATH = os.path.join(ROOT, "images_train_cropped")
OUT_PATH = ROOT
MIN_IMG_THRES = 3
TRAIN_FRAC = 0.80
QUERY_FRAC = 0.25
seed = 51221455129 % 2**32
np.random.seed(seed)

df = pd.read_csv(TRAIN_FILE_PATH)
df = df.groupby('individual_id')
df = df.filter(lambda x : len(x) >= MIN_IMG_THRES) # all individuals with more than n images

old_size = df['image'].size

# Remove all images that doesn't have crops
df = df[df['image'].apply( lambda x : os.path.exists(f"{TRAIN_IMG_PATH}/{x}"))]
print("Missing images:", old_size - df['image'].size)

# According to https://www.kaggle.com/code/andradaolteanu/whales-dolphins-effnet-embedding-cos-distance#2.-Individual-Analysis
species = {
    x: x for x in df['species'].unique()
}

species_fix = {
    "bottlenose_dolpin": "bottlenose_dolphin",
    "kiler_whale": "killer_whale",
    "beluga": "beluga_whale",
    "globis": "short_finned_pilot_whale",
    "pilot_whale": "short_finned_pilot_whale"
}

species.update(species_fix)

df['species'] = df['species'].map(species)
species_to_id = {s: i for i, s in enumerate(df['species'].unique())}
df['species_id'] = df['species'].map(species_to_id)

# Add ReId ids and Camera Ids to the df
ids = list(df['individual_id'].unique())
start_id = 0
ids_map = {iid: idx for idx, iid in enumerate(ids, start_id)}
df['camera_id'] = pd.Series([i for i in range(df.size)]).astype(int)
df['id'] = df['individual_id'].map(ids_map).astype(int)

print(df.head())

df.to_csv(f"{OUT_PATH}/all_{seed}_{MIN_IMG_THRES}.csv", index=False)

train_ids = set(random.sample(ids, int(len(ids) * TRAIN_FRAC)))
test_ids = set(ids).difference(train_ids)
print(len(train_ids), len(test_ids), train_ids.intersection(test_ids), len(ids))
print("Train attr size:", len(df['species_id'].unique()))

# Remap ids
start_id = 0
train_ids_map = {iid: idx for idx, iid in enumerate(train_ids, start_id)}
start_id = len(train_ids_map)
test_ids_map = {iid: idx for idx, iid in enumerate(test_ids, start_id)}

# Cretate train file
train_df = pd.DataFrame([], columns=df.columns)
train_df = pd.concat([df[df['individual_id'] == id] for id in train_ids])
train_df['id'] = train_df['individual_id'].map(train_ids_map).astype(int)
train_df.to_csv(f"{OUT_PATH}/train_{seed}_{MIN_IMG_THRES}.csv", index=False)

# Create query\gallery files
test_df = pd.DataFrame([], columns=df.columns)
gallery_df = pd.DataFrame([], columns=df.columns)

for id in test_ids:
    temp_df = df[df['individual_id'] == id]
    tdf = temp_df.sample(frac=QUERY_FRAC)
    gdf = temp_df.drop(tdf.index)
    test_df = pd.concat([test_df, tdf])
    gallery_df = pd.concat([gallery_df, gdf])

print("Test attr size:", len(test_df['species_id'].unique()))
print("Gallery attr size:", len(gallery_df['species_id'].unique()))

test_df['id'] = test_df['individual_id'].map(test_ids_map).astype(int)
gallery_df['id'] = gallery_df['individual_id'].map(test_ids_map).astype(int)
test_df.to_csv(f"{OUT_PATH}/test_{seed}_{MIN_IMG_THRES}.csv", index=False)
gallery_df.to_csv(f"{OUT_PATH}/gallery_{seed}_{MIN_IMG_THRES}.csv", index=False)
