from django.urls import path

from . import views

urlpatterns = [
    path('dashboard/', views.index, name='home_dashboard'),
    path('', views.index, name='index'),
    path('get-data-for-csv/',views.get_data_for_csv)
]
