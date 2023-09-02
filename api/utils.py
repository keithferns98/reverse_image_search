from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


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
