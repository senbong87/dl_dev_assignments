import os
import io
import struct
import multiprocessing as mp
from collections import defaultdict

import bson
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_bidirectional_mapping(df, col):
    forward_map, backward_map = dict(), dict()
    for idx, row in df.iterrows():
        forward_map[idx] = row[col]
        backward_map[row[col]] = idx
    return forward_map, backward_map


def read_bson(path, num_records, with_category):
    rows = dict()
    with open(path, "rb") as fhandle, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = fhandle.read(4)
            if len(item_length_bytes) == 0:
                break
                
            length = struct.unpack("<i", item_length_bytes)[0]
            fhandle.seek(offset)
            item_data = fhandle.read(length)
            assert len(item_data) == length
            
            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])
            
            row = [num_imgs, offset, length]
            if with_category:
                row += [item["category_id"]]
            rows[product_id] = row
            
            offset += length
            fhandle.seek(offset)
            pbar.update()
            
    columns = ["num_imgs", "offset", "length"]
    if with_category:
        columns += ["category_id"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def create_train_valid_datasets(train_offset_df, category_df, split_percentage=0.2,
                              drop_percentage=0.):
    cat2idx, idx2cat = create_bidirectional_mapping(category_df, "category_idx")
    
    category_pids_map = defaultdict(list)
    for pid,row in tqdm(train_offset_df.iterrows(), total=train_offset_df.shape[0]):
        category_pids_map[row["category_id"]].append(pid)
    
    train_list, valid_list = [], []
    with tqdm(total=train_offset_df.shape[0]) as pbar:
        for category_id, product_ids in category_pids_map.items():
            category_idx = cat2idx[category_id]
            
            # randomly drop products to reduce dataset
            keep_size = int(len(product_ids) * (1.0 - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)
                
            valid_size = int(len(product_ids) * split_percentage)
            if valid_size:
                valid_ids = np.random.choice(product_ids, valid_size, replace=False)
            else:
                valid_ids = []
                
            # create training & validation
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(train_offset_df.loc[product_id, "num_imgs"]):
                    if product_id in valid_ids:
                        valid_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
                
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    valid_df = pd.DataFrame(valid_list, columns=columns)
    return train_df, valid_df


if __name__ == '__main__':

    # set path variables
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    TRAIN_BSON_PATH = os.path.join(RAW_DATA_DIR, "train.bson")
    TRAIN_NUM_PRODUCTS = 7069896
    TEST_BSON_PATH = os.path.join(RAW_DATA_DIR, "test.bson")
    TEST_NUM_PRODUCTS = 1768182
    CATEGORY_PATH = os.path.join(RAW_DATA_DIR, "category_names.csv")
    TRAIN_OFFSET_PATH = os.path.join(PROCESSED_DATA_DIR, "train_offset.csv")
    TEST_OFFSET_PATH = os.path.join(PROCESSED_DATA_DIR, "test_offset.csv")
    TRAIN_IMG_PATH = os.path.join(PROCESSED_DATA_DIR, "train_img.csv")
    VALID_IMG_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_img.csv")
    TRAIN_SAMPLE_IMG_PATH = os.path.join(PROCESSED_DATA_DIR, "train_sample_img.csv")
    VALID_SAMPLE_IMG_PATH = os.path.join(PROCESSED_DATA_DIR, "valid_sample_img.csv")

    # load & preprocess data
    category_df = pd.read_csv(CATEGORY_PATH, index_col="category_id")
    category_df["category_idx"] = pd.Series(range(category_df.shape[0]), index=category_df.index)

    if not os.path.isfile(TRAIN_OFFSET_PATH):
        print("loading training bson...")
        train_offset_df = read_bson(TRAIN_BSON_PATH, TRAIN_NUM_PRODUCTS, with_category=True)
        train_offset_df.to_csv(TRAIN_OFFSET_PATH)
    else:
        train_offset_df = pd.read_csv(TRAIN_OFFSET_PATH, index_col="product_id")
        
    if not os.path.isfile(TEST_OFFSET_PATH):
        print("loading testing bson...")
        test_offset_df = read_bson(TEST_BSON_PATH, TEST_NUM_PRODUCTS, with_category=False)
        test_offset_df.to_csv(TEST_OFFSET_PATH)

    if not os.path.isfile(TRAIN_IMG_PATH) and not os.path.isfile(VALID_IMG_PATH):
        print("creating train/valid split datasets...")
        train_img_df, valid_img_df = create_train_valid_datasets(train_offset_df, category_df, split_percentage=0.002,
                                                                 drop_percentage=0.0)
        train_img_df.to_csv(TRAIN_IMG_PATH, index=False)
        valid_img_df.to_csv(VALID_IMG_PATH, index=False)

    if not os.path.isfile(TRAIN_SAMPLE_IMG_PATH) and not os.path.isfile(VALID_SAMPLE_IMG_PATH):
        print("creating sample train/valid split datasets...")
        train_sample_img_df, valid_sample_img_df = create_train_valid_datasets(train_offset_df, category_df, split_percentage=0.2,
                                                                 drop_percentage=0.99)
        train_sample_img_df.to_csv(TRAIN_SAMPLE_IMG_PATH, index=False)
        valid_sample_img_df.to_csv(VALID_SAMPLE_IMG_PATH, index=False)
