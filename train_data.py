
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import pdb
import numpy as np
import pdb
# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename #+'trainlist.txt'
        print(train_list)
        with open(train_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('rain','norain') for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]


        haze_img = Image.open(self.train_data_dir + haze_name)

        try:
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = haze_img.size
        # print(width,height,width - crop_width,height - crop_height)
        # if width < crop_width and height < crop_height :
        #     haze_img = haze_img.resize((crop_width,crop_height), Image.ANTIALIAS)
        #     gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        # elif width < crop_width :
        #     haze_img = haze_img.resize((crop_width,height), Image.ANTIALIAS)
        #     gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        # elif height < crop_height :
        #     haze_img = haze_img.resize((width,crop_height), Image.ANTIALIAS)
        #     gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        # wd_new = int(16*np.ceil(haze_img.size[0]/16.0))
        # ht_new = int(16*np.ceil(haze_img.size[1]/16.0))
        
        # width, height = haze_img.size
        # haze_img = haze_img.resize((crop_width,crop_height), Image.ANTIALIAS)
        # gt_img = gt_img.resize((crop_width,crop_height), Image.ANTIALIAS)
        # print(haze_name,width,height,width - crop_width,height - crop_height)
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        width, height = haze_crop_img.size
        # print(width,height)


        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

