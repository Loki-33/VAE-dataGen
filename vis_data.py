import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

data = np.load('rsna_dataset_128.npz')
images= data['images']
labels = data['targets']

healthy_imgs = []
p_imgs = []

i=0
while len(healthy_imgs) < 5 and len(p_imgs) < 5:
    if labels[i] == 0:
        healthy_imgs.append(images[i])
    else:
        p_imgs.append(images[i])
    i+=1

fig, axes = plt.subplots(2, 5, figsize=(15,6))
fig.suptitle('DATA VISUALIZING')

for i in range(5):
    axes[0,i].imshow(healthy_imgs[i], cmap='gray')
    axes[0,i].axis('off')
    axes[1,i].imshow(p_imgs[i], cmap='gray')
    axes[1,i].axis('off')

    if i==2:
        axes[0,i].set_title('Healthy (class 0)', fontsize=12, pad=10)
        axes[1,i].set_title('Pneumonia (class 1)', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig('ssssss.png', dpi=150, bbox_inches='tight')
plt.plot()

    

