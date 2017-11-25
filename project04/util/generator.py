import io

import bson
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, Iterator, ImageDataGenerator
from keras import backend as K


class BSONIterator(Iterator):

    def __init__(self, bson_file, image_df, offset_df, num_class, image_data_generator,
                 lock, target_size=(180, 180), with_label=True, batch_size=32,
                 shuffle=False, seed=None):
        self.file = bson_file
        self.image_df = image_df
        self.offset_df = offset_df
        self.with_label = with_label
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        self.lock = lock
        print("Found {0} images belong to {1} classes.".format(self.image_df.shape[0], self.num_class))
        super(BSONIterator, self).__init__(
            self.image_df.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_label:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        
        for i,j in enumerate(index_array):
            with self.lock:
                image_row = self.image_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offset_df.loc[product_id]
                
                # read this product's data from the BSON file
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])
                
            # grab the image from the product
            item = bson.BSON(item_data).decode()
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # preprocess the image
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # add the image and the label to the batch
            batch_x[i] = x
            if self.with_label:
                batch_y[i, image_row["category_idx"]] = 1
                    
        if self.with_label:
            return batch_x, batch_y
        else:
            return batch_x
            
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
