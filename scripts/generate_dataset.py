import pandas as pd
import numpy as np
import random

TRAIN_FILE_PATH = "/home/kmolchanov/reps/whales/data/train.csv"
TRAIN_IMG_PATH = "/home/kmolchanov/reps/whales/data/images_train_cropped"
OUT_PATH = "/home/kmolchanov/reps/whales/data/"
MIN_IMG_THRES = 3
TRAIN_FRAC = 0.80
QUERY_FRAC = 0.25
seed = 51221455129 % 2**32
np.random.seed(seed)

df = pd.read_csv(TRAIN_FILE_PATH)
df = df.groupby('individual_id')
df = df.filter(lambda x : len(x) >= MIN_IMG_THRES) # all individuals with more than n images

# Remove all images that doesn't have crops
df = df[df['image'].apply( lambda x : os.path.exists(f"{TRAIN_IMG_PATH}/{x}"))]

# Add ReId ids and Camera Ids to the df
ids = list(df['individual_id'].unique())
start_id = 0
ids_map = {iid: idx for idx, iid in enumerate(ids, start_id)}
df['camera_id'] = pd.Series([i for i in range(df.size)]).astype(int)
df['id'] = df['individual_id'].map(ids_map).astype(int)

print(df.head())

df.to_csv(f"{OUT_PATH}/all_{seed}.csv", index=False)

train_ids = set(random.sample(ids, int(len(ids) * TRAIN_FRAC)))
test_ids = set(ids).difference(train_ids)
print(len(train_ids), len(test_ids), train_ids.intersection(test_ids), len(ids))

# Remap ids
start_id = 0
train_ids_map = {iid: idx for idx, iid in enumerate(train_ids, start_id)}
start_id = len(train_ids_map)
test_ids_map = {iid: idx for idx, iid in enumerate(test_ids, start_id)}

# Cretate train file
train_df = pd.DataFrame([], columns=df.columns)
train_df = pd.concat([df[df['individual_id'] == id] for id in train_ids])
train_df['id'] = train_df['individual_id'].map(train_ids_map).astype(int)
train_df.to_csv(f"{OUT_PATH}/train_{seed}.csv", index=False)

# Create query\gallery files
test_df = pd.DataFrame([], columns=df.columns)
gallery_df = pd.DataFrame([], columns=df.columns)

for id in test_ids:
    temp_df = df[df['individual_id'] == id]
    tdf = temp_df.sample(frac=QUERY_FRAC)
    gdf = temp_df.drop(tdf.index)
    test_df = pd.concat([test_df, tdf])
    gallery_df = pd.concat([gallery_df, gdf])

test_df['id'] = test_df['individual_id'].map(test_ids_map).astype(int)
gallery_df['id'] = gallery_df['individual_id'].map(test_ids_map).astype(int)
test_df.to_csv(f"{OUT_PATH}/test_{seed}.csv", index=False)
gallery_df.to_csv(f"{OUT_PATH}/gallery_{seed}.csv", index=False)
