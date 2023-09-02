from django.urls import path
from api.views import GenSimilarImagesAPIView

urlpatterns = [
    path(
        "upload/",
        GenSimilarImagesAPIView.as_view(),
        name="Preprocess and gen similar images",
    ),
]
