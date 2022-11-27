import os
import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from model.SegModel import seg
import torch.nn.functional as F
import pydicom
from medpy.metric.binary import dc, hd, asd,jc


modelPath=".../modelWeight//modelWeight.pth"

model = seg()
model.load_state_dict(torch.load(modelPath))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testFilePath=".../testFold.txt"


def dice_coefficient(y_true, y_pred, smooth=0.00001):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    smooth = 1
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f ) + smooth)


def readDS(imgPath):
    cine_ds = pydicom.read_file(imgPath, force=True)
    cine_ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
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

def preprocess(imgPath):
    imgDS=readDS(imgPath)
    imgMatrix=imgDS.pixel_array
    space=imgDS.PixelSpacing
    cropMask=crop_or_pad_slice_to_size(imgMatrix,256,256)
    if len(cropMask.shape) == 2:
        cropMask = np.expand_dims(cropMask, axis=2)
    return cropMask


patientList=[]

with open(testFilePath) as f:
    allPath=f.readlines()

for i in allPath:
    patient=i.split("/")[7]
    if patient not in patientList:
        patientList.append(patient)


print(len(allPath))
print(patientList)
phaseList=["ph0001","ph0002","ph0003","ph0004","ph0005","ph0006","ph0007","ph0008","ph0009","ph0010","ph0011","ph0012","ph0013","ph0014","ph0015",
             "ph0016","ph0017","ph0018","ph0019","ph0020","ph0021","ph0022","ph0023","ph0024","ph0025","ph0026","ph0027","ph0028","ph0029","ph0030"]

dcList=[]
jcList=[]
hdList=[]
veList=[]
lvefList=[]
num=0
case=0


for p in patientList:
    patientDC=[]
    patientJC=[]
    patientVolume=[]
    volumeGtList=[]
    volumePreList=[]
    for ph in phaseList:
        testPath=[]
        for path in allPath:
            if p in path and ph in path:
                testPath.append(path)

        prediction=np.zeros((256,256,len(testPath)))
        groundTruth=np.zeros((256,256,len(testPath)))

        for n in range(len(testPath)):
            i=testPath[n]
            mask_file = i.split('\n')[0]
            mags_file = mask_file.replace("SAX4DFMASK", "SAX4DFMAG")
            vx_file = mask_file.replace("SAX4DFMASK", "SAX4DFX")
            vy_file = mask_file.replace("SAX4DFMASK", "SAX4DFY")
            vz_file = mask_file.replace("SAX4DFMASK", "SAX4DFZ")

            mags = preprocess(mags_file)
            vx = preprocess(vx_file)
            vy = preprocess(vy_file)
            vz = preprocess(vz_file)


            vel = np.append(vx, vy, axis=-1)
            vel = np.append(vel, vz, axis=-1)

            vel = np.transpose(vel, (-1, 1, 0))
            vel = np.expand_dims(vel,axis=0)

            img = normalization(mags)
            img = np.transpose(img,(-1,1,0))
            img = np.expand_dims(img, axis=1)

            mask = preprocess(mask_file)
            mask = np.transpose(mask, (-1, 1, 0))
            mask = ((mask == 1)) * 1
            groundTruth[:,:,n]=mask[0]

            testDataset = torch.from_numpy(img)
            velDataset = torch.from_numpy(vel)

            imgDS = readDS(vx_file)
            space = imgDS.PixelSpacing
            #
            with torch.no_grad():
                model.to(device)
                #
                output = model(testDataset.to(device, dtype=torch.float),velDataset.to(device, dtype=torch.float))
            pred = (output > 0.5).float()
            pred = pred.cpu()
            pred = pred.numpy()

            prediction[:, :, n] = pred[0, 0]


        volumeGT = np.sum(groundTruth) * space[0] * space[1] * 3
        volumePre = np.sum(prediction) * space[0] * space[1] * 3
        vErr = abs(volumeGT - volumePre) / volumeGT
        volumeGtList.append(volumeGT)
        volumePreList.append(volumePre)

        dice = dc(prediction, groundTruth)
        hsd = asd(prediction, groundTruth, (space[0], space[1], 3))
        dcList.append(dice)
        veList.append(vErr)
        print(p, ph, dice, hsd)

        hdList.append(hsd)
        jcs = jc(prediction, groundTruth)
        jcList.append(jcs)
        patientDC.append(dice)
        patientJC.append(jcs)
        patientVolume.append(vErr)


    ed = max(volumeGtList)
    es = min(volumeGtList)


    lvefGt = (ed - es) / ed
    edphase = volumeGtList.index(ed)
    esphase = volumeGtList.index(es)
    lvefPre = (volumePreList[edphase] - volumePreList[esphase]) / volumePreList[edphase]
    lvef_err = abs(lvefGt - lvefPre)
    lvefList.append(lvef_err)


print(np.mean(dcList))
print(np.mean(jcList))
print(np.mean(hdList))
print(np.mean(veList))
print(np.mean(lvefList))