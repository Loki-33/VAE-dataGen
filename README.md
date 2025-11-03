# VAE-DataGen 

Synthetic data generation using CNN-conditionalVAE. <br>

Dataset was from kaggle rsna-pneumonia 
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge <br>
The file was of format DICOM, so we had to use pydicom to take out relevant data and then create a dataset using 'convert.py'.


TEST RESULT:<br>
Total Loss: 93.0469<br>
Reconstruction Loss: 63.8645<br>
KL Divergence: 58.3649<br>
SSIM: 0.7155<br>
PSNR: 24.40 dB<br>
