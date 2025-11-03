# VAE-DataGen 

Synthetic data generation using CNN-conditionalVAE. <br>

Dataset was from kaggle rsna-pneumonia 
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge
The file was of format DICOM, so we had to use pydicom to take out relevant data and then create a dataset using 'convert.py'.


TEST RESULT:
Total Loss: 93.0469
Reconstruction Loss: 63.8645
KL Divergence: 58.3649
SSIM: 0.7155
PSNR: 24.40 dB
