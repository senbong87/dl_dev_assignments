from __future__ import print_function
import io

import bson
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.xception import preprocess_input
from keras import backend as K
from tqdm import tqdm

from util.generator import BSONIterator
from util.preprocess import create_bidirectional_mapping


# load model
model = load_model(".model_checkpts/weights.44-1.53.hdf5")

# global variable
NUM_TEST_PRODUCTS = 1768182

# load data
category_df = pd.read_csv("data/raw/category_names.csv", index_col="category_id")
category_df["category_idx"] = pd.Series(range(category_df.shape[0]), index=category_df.index)
submission_df = pd.read_csv("data/raw/sample_submission.csv")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
data = bson.decode_file_iter(open("data/raw/test.bson", "rb"))

# create category to index mapping and index to category mapping
cat2idx, idx2cat = create_bidirectional_mapping(category_df, "category_idx")

# prepare submission file
with tqdm(total=NUM_TEST_PRODUCTS) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])
        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]        
        pbar.update()

# output submission file
submission_df.to_csv("submission.csv.gz", compression="gzip", index=False)
