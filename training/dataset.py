from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import pydicom


def readDS(imgPath):
    cine_ds = pydicom.read_file(imgPath, force=True)
    cine_ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    # pix_space=cine_ds.PixelSpacing
    # pix = cine_ds.pixel_array
    return cine_ds

def crop_or_pad_slice_to_size(img, nx, ny):

    x, y = img.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]

    return slice_cropped
def normalization(array):
    maxValue = np.percentile(array, 95, interpolation='nearest')
    minValue = np.percentile(array, 5, interpolation='nearest')
    re_imgArray = (array - minValue) / (maxValue - minValue + 0.000000001)
    return re_imgArray

def normVel(array):
    maxValue = np.percentile(array, 95, interpolation='nearest')
    minValue = np.percentile(array, 5, interpolation='nearest')
    re_imgArray = (array - minValue) / (maxValue - minValue + 0.000000001)
    return re_imgArray


class BasicDataset(Dataset):
    # def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
    def __init__(self, filePath):


        self.ids=open(filePath).readlines()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, imgPath):
        imgDS=readDS(imgPath)
        imgMatrix=imgDS.pixel_array
        space=imgDS.PixelSpacing
        cropMask=crop_or_pad_slice_to_size(imgMatrix,256,256)


        if len(cropMask.shape) == 2:
            cropMask = np.expand_dims(cropMask, axis=2)

        # # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255

        return cropMask

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = idx.split('\n')[0]
        mags_file = mask_file.replace("SAX4DFMASK","SAX4DFMAG")
        # print(mask_file)
        vx_file=mask_file.replace("SAX4DFMASK","SAX4DFX")
        vy_file=mask_file.replace("SAX4DFMASK","SAX4DFY")
        vz_file=mask_file.replace("SAX4DFMASK","SAX4DFZ")


        mags = self.preprocess(mags_file)
        mags = normalization(mags)
        mags = np.transpose(mags,(-1,1,0))


        vx = self.preprocess(vx_file)
        vx = np.transpose(vx, (-1, 1, 0))
        vx = normalization(np.absolute(vx))

        vy = self.preprocess(vy_file)
        vy = np.transpose(vy, (-1, 1, 0))
        vy = normalization(np.absolute(vy))

        vz = self.preprocess(vz_file)
        vz = np.transpose(vz, (-1, 1, 0))
        vz = normalization(np.absolute(vz))

        vel = np.append(vx, vy, axis=0)
        vel = np.append(vel, vz, axis=0)

        # img=np.append(mags,flow,axis=0)
        # print("dataset",img.shape)
        mask = self.preprocess(mask_file)
        mask=((mask==1))*1

        mask=np.transpose(mask,(-1,1,0))

        # print(np.shape(mask))
        return {
            'image': torch.from_numpy(mags).type(torch.FloatTensor),
            'vel': torch.from_numpy(vel).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
