import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = np.load('rsna_dataset_128.npz')

images = dataset['images']
targets = dataset['targets']

x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)/255.0
y = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(x, y)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=64, num_classes=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes


        self.encoder = nn.Sequential(
              nn.Conv2d(1, 32, 4, 2,1),
              nn.ReLU(),
              nn.Conv2d(32, 64, 4, 2,1),
              nn.ReLU(),
              nn.Conv2d(64, 128, 4, 2,1),
              nn.ReLU(),
              nn.Conv2d(128, 256, 4, 2,1),
              nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256*8*8+num_classes, latent_dim)
        self.fc_logvar = nn.Linear(256*8*8+num_classes, latent_dim)


        self.fc_decoder = nn.Linear(latent_dim+num_classes, 256*8*8)
        self.decoder = nn.Sequential(
              nn.ConvTranspose2d(256, 128, 4, 2, 1),
              nn.ReLU(),
              nn.ConvTranspose2d(128, 64, 4, 2, 1),
              nn.ReLU(),
              nn.ConvTranspose2d(64, 32, 4, 2, 1),
              nn.ReLU(),
              nn.ConvTranspose2d(32, 1, 4, 2, 1),
              nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y):
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        h = self.encoder(x)
        h = self.flatten(h)
        h = torch.cat([h, y_onehot], dim=1)

        mu=self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
 
        z = torch.cat([z, y_onehot], dim=1)
        out = self.fc_decoder(z)
        out = out.view(-1, 256, 8, 8)
        out = self.decoder(out)
        return out, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())

    total_loss = (recon_loss + beta * kl_loss)/x.size(0)

    return total_loss, recon_loss/x.size(0), kl_loss/x.size(0)

model = ConditionalVAE()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

def train():
  model.train()
  for epoch in range(20):
      total_loss = 0
      total_kl = 0
      total_recon = 0

      for x, y in train_loader:
        recon, mu, logvar = model(x,y)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=0.5)
        optimizer.zero_grad()
        loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimizer.step()
      total_kl += kl_loss.item()
      total_recon += recon_loss.item()
      total_loss += loss.item()
        
      avg_loss = total_loss / len(train_loader)
      avg_recon = total_recon / len(train_loader)
      avg_kl = total_kl / len(train_loader)

      print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

def test():
  model.eval()
  total_loss = 0
  total_recon = 0
  total_kl = 0

  from skimage.metrics import structural_similarity as ssim
  ssim_scores = []
  psnr_scores = []

  with torch.no_grad():
    for x, y in test_loader:
      recon, mu, logvar = model(x, y)
      loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=0.5)
      total_loss += loss.item()
      total_recon += recon_loss.item()
      total_kl += kl_loss.item()

      for i in range(x.size(0)):
        orig = x[i, 0].cpu().numpy()
        rec = recon[i, 0].cpu().numpy()

        ssim_val = ssim(orig, rec, data_range=1.0)
        ssim_scores.append(ssim_val)

        mse = np.mean((orig-rec)**2)
        if mse > 0:
          psnr = 20 * np.log10(1.0 / np.sqrt(mse))
          psnr_scores.append(psnr)

  avg_loss = total_loss / len(test_loader)
  avg_recon = total_recon / len(test_loader)
  avg_kl = total_kl / len(test_loader)
  avg_ssim = np.mean(ssim_scores)
  avg_psnr = np.mean(psnr_scores)

  print(f"\n=== Test Results ===")
  print(f"Total Loss: {avg_loss:.4f}")
  print(f"Reconstruction Loss: {avg_recon:.4f}")
  print(f"KL Divergence: {avg_kl:.4f}")
  print(f"SSIM: {avg_ssim:.4f}")
  print(f"PSNR: {avg_psnr:.2f} dB")

'''
Reconstruction Loss (MSE/SSIM) - How well it reconstructs test images

Structural Similarity Index (SSIM) - Perceptual quality by comparing
comparing luminance, contrast, and structure. 
SSIM = [luminance_similarity] × [contrast_similarity] × [structure_similarity

Peak Signal-to-Noise Ratio (PSNR) - Image quality metric
PSNR = 20 * log₁₀(MAX_PIXEL_VALUE / √MSE) 
higher better, measures image quality in decibels

'''

if __name__ == '__main__':
    # train()
    # torch.save(model, 'model.pth')

    model = torch.load('model.pth', weights_only=False)
    test()


'''
TEST RESULT:

Total Loss: 93.0469
Reconstruction Loss: 63.8645
KL Divergence: 58.3649
SSIM: 0.7155
PSNR: 24.40 dB

'''