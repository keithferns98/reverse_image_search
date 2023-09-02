from django.db import models

# Create your models here.
class StoreSimilarImages(models.Model):
    base_image = models.CharField(max_length=150)
    similar_images = models.CharField(max_length=200)

    def __str__(self):
        return self.base_image

