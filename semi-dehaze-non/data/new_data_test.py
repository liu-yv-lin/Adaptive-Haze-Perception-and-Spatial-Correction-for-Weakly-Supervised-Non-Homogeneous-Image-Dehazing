import torch.utils.data as data
import os
import os.path
import numpy as np
import glob
import cv2
import random
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.h5'
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def is_jpg_file(filename):
  return filename.endswith('.jpg')

def make_dataset(dirs):
  images = []
  if isinstance(dirs, list):
     for i, dir in enumerate(dirs):
        if not os.path.isdir(dir):
            raise Exception('Check dataroot')
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(dir, fname)
                    item = path
                    images.append(item)
  else:
      for root, _, fnames in sorted(os.walk(dirs)):
          for fname in fnames:
              if is_image_file(fname):
                  path = os.path.join(root, fname)
                  images.append(path)
  return images

def get_patch(img_in, img_tar,trans_map, patch_size):
    ih, iw = img_in.shape[:2]
    p = patch_size
    ix = random.randrange(0, iw - p + 1)
    iy = random.randrange(0, ih - p + 1)
    img_in = img_in[iy:iy + p, ix:ix + p, :]
    img_tar = img_tar[iy:iy + p, ix:ix + p, :]
    trans_map = trans_map[iy:iy + p, ix:ix + p, :]
    return img_in, img_tar, trans_map

class new_data(data.Dataset):
  def __init__(self, opt, dataroot, seed=None):

    self.dir = dataroot
    self.transform_flip = opt.flip
    self.size = 288
    if seed is not None:
      np.random.seed(seed)
    self.input = sorted(make_dataset(os.path.join(self.dir, 'haze')))
    self.GT = os.path.join(self.dir, 'gt')
    self.trans = os.path.join(self.dir, 'gt')

  def __getitem__(self, index):
      input_path = self.input[index]
      filename = os.path.basename(input_path)
      GT_path = os.path.join(self.GT, filename)
      trans_path = os.path.join(self.GT, filename)
      haze_image = cv2.imread(input_path).astype("float") / 255.0
      GT = cv2.imread(GT_path).astype("float") / 255.0
      trans = cv2.imread(trans_path).astype("float") / 255.0


      haze_image, GT, trans = self.transform(haze_image, GT, trans)
      cv2.setNumThreads(0)
      cv2.ocl.setUseOpenCL(False)

      gt = [ haze_image.astype(np.float32), GT.astype(np.float32), trans.astype(np.float32)]
      return gt
  def __len__(self):
    train_lb_list=glob.glob(os.path.join(self.dir, 'haze')+'/*jpg')
    return len(train_lb_list)

  def transform(self, haze_image, GT, trans_map):
       haze_image = np.swapaxes(haze_image, 0, 2)
       trans_map = np.swapaxes(trans_map, 0, 2)
       GT = np.swapaxes(GT, 0, 2)

       haze_image = np.swapaxes(haze_image, 1, 2)
       trans_map = np.swapaxes(trans_map, 1, 2)
       GT = np.swapaxes(GT, 1, 2)
       return haze_image, GT, trans_map

  def padding(self, haze_image):
        h_gt = haze_image.shape[0]
        w_gt = haze_image.shape[1]
        h = int(haze_image.shape[0] / 16 + 1) * 16
        w = int(haze_image.shape[1] / 16 + 1) * 16
        haze_image = cv2.copyMakeBorder(haze_image, 0, h - int(haze_image.shape[0]), 0,
                                           w - int(haze_image.shape[1]), cv2.BORDER_REPLICATE)
        return haze_image, h_gt, w_gt