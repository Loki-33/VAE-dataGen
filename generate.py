import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from main import ConditionalVAE

model = torch.load('model.pth', weights_only=False)
model.eval()


def generate_images(model, num_samples=5, class_label=0):
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        y = torch.tensor([class_label] * num_samples, dtype=torch.long)
        y_onehot = F.one_hot(y, num_classes=model.num_classes).float()


        z_cond = torch.cat([z, y_onehot], dim=1)
        out = model.fc_decoder(z_cond)
        out = out.view(-1, 256, 8, 8)
        generated_images = model.decoder(out)

    return generated_images


def vis():
    healthy_imgs = generate_images(model, num_samples=5, class_label=0)
    p_imgs = generate_images(model, num_samples=5, class_label=1)

    fig, axes = plt.subplots(2, 5, figsize=(15,6))
    fig.suptitle('Genreated images')


    for i in range(5):
        img = healthy_imgs[i, 0].numpy()
        axes[0,i].imshow(img, cmap='gray')
        axes[0,i].axis('off')
        if i==2:
            axes[0,i].set_title('Healthy (class 0)', fontsize=12, pad=10)


    for i in range(5):
        img = p_imgs[i, 0].numpy()
        axes[1,i].imshow(img, cmap='gray')
        axes[1,i].axis('off')
        if i==2:
            axes[1,i].set_title('Pneumonia (Class 1)', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig('generated_xrays.png', dpi=150, bbox_inches='tight')
    

def save_individual_images(num_each=10):
    """
    Save individual generated images to files
    """
    import os
    os.makedirs('generated', exist_ok=True)
    
    print(f"Generating {num_each} images of each class...")
    
    #  healthy
    healthy_imgs = generate_images(model, num_samples=num_each, class_label=0)
    for i in range(num_each):
        img = (healthy_imgs[i, 0].numpy() * 255).astype(np.uint8)
        plt.imsave(f'generated/healthy_{i:03d}.png', img, cmap='gray')
    
    # pneumonia
    pneumonia_imgs = generate_images(model, num_samples=num_each, class_label=1)
    for i in range(num_each):
        img = (pneumonia_imgs[i, 0].numpy() * 255).astype(np.uint8)
        plt.imsave(f'generated/pneumonia_{i:03d}.png', img, cmap='gray')
    
    print(f"Saved {num_each*2} images to 'generated/' directory")


if __name__ == '__main__':
    print("\n1. Generating sample images...")
    vis()
    print("\n2. Saving individual samples....")
    save_individual_images()
    print("\nDone")


