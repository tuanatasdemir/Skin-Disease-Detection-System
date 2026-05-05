import torch
import cv2
import numpy as np
import os
import glob
from src.model_pytorch import SkinCNN
from src.analysis_tools import detect_blemishes_by_color, detect_texture_issues
from torchvision import transforms
from PIL import Image

DATA_PATH = r"data\processed"
MODEL_PATH = r"outputs\models\skin_model_pytorch.pth"
CLASSES = ['Acne', 'Moles', 'Healthy', 'Rosacea', 'Vitiligo', 'Actinic_Keratosis'] 

num_classes = len(CLASSES)
model = SkinCNN(num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def run_batch_analysis(base_path, num_samples=2):
    """Her kategoriden belirli sayıda örnek resmi analiz eder."""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"{'Dosya':<20} | {'CNN Teşhisi':<15} | {'Sivilce Yoğunluğu':<15} | {'Doku Çizgisi'}")
    print("-" * 75)

    for category in CLASSES:
        search_path = os.path.join(base_path, "**", category, "*.jpg")
        image_list = glob.glob(search_path, recursive=True)[:num_samples]

        for img_path in image_list:
            _, color_mask, _ = detect_blemishes_by_color(img_path)
            _, _, edges = detect_texture_issues(img_path)
            
            blemish_score = np.sum(color_mask > 0)
            edge_score = np.sum(edges > 0)

            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                cnn_result = CLASSES[predicted.item()]

            file_name = os.path.basename(img_path)
            print(f"{file_name:<20} | {cnn_result:<15} | {blemish_score:<15} | {edge_score}")

if __name__ == "__main__":
    run_batch_analysis(DATA_PATH)