from PIL import Image
import os

def resize_images(inputFolder,outputFolder):
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for filename in os.listdir(inputFolder):
        image_path = os.path.join(inputFolder,filename)
        print("image_path       ",image_path)
        output_path = os.path.join(outputFolder,filename)

        image = Image.open(image_path)
        width,height = image.size

        new_width = width-(width%16)
        new_height = height-(height%16)

        resize_image = image.resize((new_width,new_height))
        resize_image.save(output_path)
        print(output_path)

input_folder = "dataset/O-HAZE/scale_8/gt"
output_folder = "dataset/O-HZAE/scale_8/gt"
resize_images(input_folder,output_folder)