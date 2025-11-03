import numpy as np
import os
from tqdm import tqdm 
import pydicom
import cv2
import pandas as pd
'''
DCM(DICOM) - Digital Imaging and Communications in Medicine
common format for medical images such as x-rays, CT scans, MRI scans
contains: 
1) image data 

2) Meta-data : 
patient name, id, age, sex
date, time 
'''
input_dir = "Dataset/stage_2_train_images"
labels_df = pd.read_csv('Dataset/stage_2_train_labels.csv')

labels = dict(zip(labels_df['patientId'], labels_df['Target']))

images = []
targets = []

for file in tqdm(os.listdir(input_dir)):
    if file.endswith('.dcm'):
        patient_id = file.replace('.dcm', '')
        if patient_id not in labels:
            continue
        path = os.path.join(input_dir, file)
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        img -= np.min(img)
        img /= np.max(img)
        img *= 255.0
        img = img.astype(np.uint8)
        img = cv2.resize(img, (128,128))
        images.append(img)
        targets.append(labels[patient_id])

images = np.array(images)[...,np.newaxis]
targets = np.array(targets)

np.savez_compressed('rsna_dataset_128.npz', images=images, targets=targets)
