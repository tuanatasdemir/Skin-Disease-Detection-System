import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from src.model_pytorch import SkinCNN
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    full_dataset = datasets.ImageFolder(root='data/processed/Acne', transform=transform)
except FileNotFoundError:
    print("Uyarı: Geçersiz dosyalar temizleniyor veya yol kontrol ediliyor...")
    full_dataset = datasets.ImageFolder(root='data/processed', transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(full_dataset.classes)
model = SkinCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Eğitim başlıyor... Toplam kategori: {num_classes}")
epochs = 5 

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Kayıp (Loss): {running_loss/len(train_loader):.4f}")

os.makedirs('outputs/models', exist_ok=True)
torch.save(model.state_dict(), 'outputs/models/skin_model_pytorch.pth')
print("Model başarıyla kaydedildi: outputs/models/skin_model_pytorch.pth")