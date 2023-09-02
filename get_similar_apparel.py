import weaviate
import csv
import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import argparse


class EfficientNetModel:
    def __init__(self):
        self.efficientnet_b7 = EfficientNetB7(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

    def load_model(self):
        efficient_model = self.efficientnet_b7.output
        res_embeddings = GlobalAveragePooling2D()(efficient_model)
        resembedding_model = Model(
            inputs=self.efficientnet_b7.input, outputs=res_embeddings
        )
        return resembedding_model


class GenDatasets:
    def load_datasets(self):
        datasets = []
        data_dir = os.path.join(
            os.getcwd(), "frontend", "public", "Datasets_for_sample"
        )
        for curr_dir in sorted(os.listdir(data_dir), key=lambda x: x[0]):
            datasets.extend(
                [
                    os.path.join(data_dir, curr_dir, filename)
                    for filename in sorted(
                        os.listdir(os.path.join(data_dir, curr_dir)),
                        key=lambda x: int(re.search(r"(\d+)\.[a-z]+$", x).group(1)),
                    )
                ]
            )
        return datasets

    def data_train_test_split(self):
        fash_data = self.load_datasets()
        X_train, y_train = train_test_split(
            fash_data, train_size=0.90, test_size=0.10, random_state=29
        )
        return X_train, y_train


class BuildSchema:
    def create_schema(self):
        schema = "FashionSimilarity"
        if self.client.schema.exists(schema):
            self.client.schema.delete_class(schema)
        schema_obj = {
            "class": "FashionSimilarity",
            "description": "Store all the image embeddings and find similar images given query image.",
            "vectorIndexConfig": {
                "distance": "cosine",
            },
            "vectorizer": "none",
            "properties": [
                {
                    "name": "img_path",
                    "dataType": ["text"],
                },
            ],
        }
        self.client.schema.create_class(schema_obj)


class ReverseImageSearch(EfficientNetModel, GenDatasets, BuildSchema):
    def __init__(self):
        super().__init__()
        self.X_train, self.y_train = self.data_train_test_split()
        self.client = weaviate.Client("http://localhost:8080")

    def preprocess_image_to_array(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        x = preprocess_input(img_array)
        return x

    def create_embeddings(self):
        model = self.load_model()
        embeddings = {}
        c = 0
        for curr_path in self.X_train:
            x = self.preprocess_image_to_array(curr_path)
            predictions = model.predict(x)
            embeddings[curr_path] = predictions[0]
            c += 1
            print(c)
        return embeddings

    def add_embeds_to_weaviate(self):
        print("Ingesting to weaviate.")
        embeddings = self.create_embeddings()
        self.client.batch.configure(batch_size=10)
        with self.client.batch as batch:
            for curr_vec in embeddings:
                img_path = curr_vec
                ebd = embeddings[curr_vec]
                batch_data = {"img_path": img_path}
                batch.add_data_object(
                    data_object=batch_data, class_name="FashionSimilarity", vector=ebd
                )
        print("Ingested to weaviate")

    def quantify_similarities(self, k=5):
        output = {}
        count = 0
        # print(len(self.X_train), len(self.y_train))
        for img_path in self.y_train:
            if os.path.exists(img_path):
                model = self.load_model()
                x = self.preprocess_image_to_array(img_path)
                dense_vec = model.predict(x)
                vec = {"vector": dense_vec[0]}
                result = (
                    self.client.query.get(
                        "FashionSimilarity", ["img_path", "_additional {certainty}"]
                    )
                    .with_near_vector(vec)
                    .with_limit(k)
                    .do()
                )

                closest_images = result.get("data").get("Get").get("FashionSimilarity")
                if not output.get(img_path):
                    output[img_path] = {"pred_paths": []}
                    output[img_path]["pred_paths"].extend(
                        [
                            curr_p.get("img_path")
                            for curr_p in closest_images
                            if curr_p.get("img_path") != img_path
                        ]
                    )
                count += 1
                print(count)
        schema_db_counts = (
            self.client.query.aggregate("FashionSimilarity")
            .with_meta_count()
            .do()["data"]["Aggregate"]["FashionSimilarity"][0]["meta"]["count"]
        )
        return output, schema_db_counts

    def top_k_precision(self, top_k=350):
        result, schema_db_counts = self.quantify_similarities()
        for curr_res in result:
            quered_cat = os.path.basename(curr_res).split("_")[0]
            c = 0
            for pred_img in result[curr_res]["pred_paths"]:
                pred_cat = os.path.basename(pred_img).split("_")[0]
                if pred_cat == quered_cat:
                    c += 1

            result[curr_res]["predictions"] = c / len(result[curr_res])
        print(result)
        # sorted_pred=sorted(result.items(), key=lambda x: x[1]["predictions"], reverse=True)
        precision_top_k = (
            len(
                [
                    result[curr_res]
                    for curr_res in result
                    if result[curr_res]["predictions"] > 3
                ]
            )
            / top_k
        )
        percentage = "{:.0f}%".format(precision_top_k * 100)
        with open("summary_top_k_precision_report.csv", "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(
                ["Top_k_precision_test_data", "percentage", "Total_X_train_records"]
            )
            csv_writer.writerow(
                [
                    f"Top_k_precision: {top_k} and test_data: {len(self.y_train)}",
                    percentage,
                    schema_db_counts,
                ]
            )


if __name__ == "__main__":
    r = ReverseImageSearch()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ingestion",
        default=0,
        nargs="?",
        choices=[0, 1],
        help="Enter 1 to do the ingestion else 0 to skip the ingestion.",
        type=int,
    )
    # parser.add_argument(
    #     "image_path", help="Enter the image path to generate similar images", type=str
    # )
    args = vars(parser.parse_args())
    print(args)
    if args["ingestion"]:
        r.create_schema()
        r.add_embeds_to_weaviate()
    r.top_k_precision()
    # if args["image_path"]:
    # print(r.quantify_similarities())
