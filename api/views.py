from django.shortcuts import render
from rest_framework.views import APIView, Response
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
import PIL
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from api.utils import EfficientNetModel
from tensorflow.keras.preprocessing import image
import weaviate
from core.models import StoreSimilarImages
import os

# Create your views here.


class GenSimilarImagesAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_name = request.data["files"].__dict__["_name"]
        output = {}
        img1 = request.FILES.getlist("files", "None")
        img = Image.open(img1[0])
        resized_image = img.resize((224, 224))
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_image = preprocess_input(img_array)
        model = EfficientNetModel()
        efficient_model = model.load_model()
        dense_vec = efficient_model.predict(preprocessed_image)
        client = weaviate.Client("http://localhost:8080")
        sim_images = StoreSimilarImages.objects.filter(base_image=file_name)
        if len(sim_images) > 1:
            final_results = []
            for curr_obj in sim_images:
                final_results.append(curr_obj.similar_images)
        else:
            vec = {"vector": dense_vec[0]}
            result = (
                client.query.get(
                    "FashionSimilarity", ["img_path", "_additional {certainty}"]
                )
                .with_near_vector(vec)
                .with_limit(6)
                .do()
            )
            print(result)
            closest_images = result.get("data").get("Get").get("FashionSimilarity")
            if not output.get(file_name):
                output[file_name] = {"pred_paths": []}
                output[file_name]["pred_paths"].extend(
                    [
                        curr_p.get("img_path")
                        for curr_p in closest_images
                        if os.path.basename(curr_p.get("img_path")) != file_name
                    ]
                )
            final_results = next(iter(output.values()))["pred_paths"]
            for val in next(iter(output.values()))["pred_paths"]:
                image_similar = StoreSimilarImages()
                image_similar.base_image = file_name
                image_similar.similar_images = val
                image_similar.save()
        print({"file_name": file_name, "results": final_results})
        return Response({"file_name": file_name, "results": final_results})
