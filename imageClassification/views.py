from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from django.views.generic import View
import cv2 as cv
from django.core.files.storage import FileSystemStorage
import urllib
import base64

    
    #cv2.imshow('Image', image)
    #cv2.waitKey(0)

def index(request):

    if request.method == "POST":
        class_names = ['apple', 'quarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
               'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        model = tf.keras.models.load_model(
            'imageClassification/nn.h5', custom_objects=None, compile=True, options=None
        )

        req = urllib.request.urlopen(request.POST['url'])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv.imdecode(arr, -1) # 'Load it as it is'
        
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        width = int(32)
        height = int(32)
        dim = (width, height)
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)


        prediction = model.predict(np.array([img]) / 255)

        return  render(request, 'images/imageProcessed.html', {
            'prediction': class_names[np.argmax(prediction)],
        })


    return  render(request, 'images/index.html')

# class index(View):
#     def get(self):
#         return render(request, 'images/index.html')

#     def post(self):
#         return render(request, 'images/index.html')

