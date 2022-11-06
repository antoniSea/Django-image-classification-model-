from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('categories', views.categories, name='categories'),
    path('api/docs', views.apiDoscs, name='apiDocs'),
    path('api', views.api, name='api')
    
]