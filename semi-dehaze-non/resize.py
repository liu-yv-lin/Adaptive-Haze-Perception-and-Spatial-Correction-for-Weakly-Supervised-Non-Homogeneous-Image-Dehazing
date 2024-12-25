import os
from PIL import Image
from PIL.Image import Resampling

dir_img="dataset/O-HAZE/scale_8/gt"
dir_save= "dataset/O-HAZE/scale_16/gt"
size=(544,400)

list_temp = os.listdir(dir_img)
for img_name in list_temp:
    img_path=dir_img+"/"+img_name
    print(img_path)
    old_img=Image.open(img_path)
    save_path=dir_save+"/"+img_name
    print(save_path)

    old_img.resize(size,Resampling.LANCZOS).save(save_path)