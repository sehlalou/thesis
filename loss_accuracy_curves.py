import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data
df = pd.read_csv('/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models/study_vit/epoch_metrics.csv')
epochs = df['epoch']

# Plotting the loss curves
plt.figure(figsize=(8, 6))
plt.plot(epochs, df['train_loss'], marker='o', linestyle='-', label='Training Loss')
plt.plot(epochs, df['val_loss'], marker='s', linestyle='--', label='Validation Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=16)
plt.xticks(epochs)  # Integer ticks only
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.close()

# Plotting the accuracy curves
plt.figure(figsize=(8, 6))
plt.plot(epochs, df['train_accuracy'], marker='o', linestyle='-', label='Training Accuracy')
plt.plot(epochs, df['val_accuracy'], marker='s', linestyle='--', label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training and Validation Accuracy', fontsize=16)
plt.xticks(epochs)  # Integer ticks only
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=300)
plt.close()

print("Plotted")
