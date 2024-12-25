import torch
import os
from PIL import  Image

def save_many_image(idx,image_tensor,image_name,file_path):
    image_tensor = image_tensor.clone().detach()
    image_tensor = image_tensor.to(torch.device("cpu"))
    image_tensor = image_tensor.squeeze()
    image_tensor = image_tensor.permute(1, 2, 0).type(torch.uint8).numpy()

    savepath = os.path.join(file_path, image_name)
    dehaze_img = Image.fromarray(image_tensor)
    dehaze_img.save(savepath)