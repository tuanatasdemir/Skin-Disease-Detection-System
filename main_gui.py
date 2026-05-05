import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import os
import random
from src.model_pytorch import SkinCNN
from src.analysis_tools import detect_blemishes_by_color, detect_texture_issues, get_advanced_features
from torchvision import transforms

class SkinAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NKU Cilt Analizi Sistemi - Çok Teknikli Panel")
        self.root.geometry("1200x850")
        self.root.configure(bg="#f8f9fa")

        self.data_path = "data/processed"
        self.classes = sorted([d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))])

        self.model = SkinCNN(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load('outputs/models/skin_model_pytorch.pth'))
        self.model.eval()

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(self.root, text="Cilt Sorunu Tespit ve Analiz Paneli", 
                         font=("Segoe UI", 22, "bold"), bg="#f8f9fa", fg="#2c3e50")
        title.pack(pady=10)

        ctrl_frame = tk.Frame(self.root, bg="#f8f9fa")
        ctrl_frame.pack(pady=10)

        self.cat_combo = ttk.Combobox(ctrl_frame, values=self.classes, state="readonly", font=("Segoe UI", 11))
        self.cat_combo.grid(row=0, column=0, padx=5)
        if self.classes: self.cat_combo.current(0)

        self.btn_analyze = tk.Button(ctrl_frame, text="Rastgele Örnek Analiz Et", command=self.analyze_random,
                                     font=("Segoe UI", 11, "bold"), bg="#27ae60", fg="white", padx=15)
        self.btn_analyze.grid(row=0, column=1, padx=10)

        img_container = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid")
        img_container.pack(pady=10, padx=20)

        f1 = tk.Frame(img_container, bg="white")
        f1.grid(row=0, column=0, padx=20, pady=10)
        self.lbl_orig = tk.Label(f1, bg="white")
        self.lbl_orig.pack()
        tk.Label(f1, text="1. Orijinal Görüntü", font=("Segoe UI", 10, "bold"), bg="white", fg="#34495e").pack()

        f2 = tk.Frame(img_container, bg="white")
        f2.grid(row=0, column=1, padx=20, pady=10)
        self.lbl_enhanced = tk.Label(f2, bg="white")
        self.lbl_enhanced.pack()
        tk.Label(f2, text="2. İyileştirme (CLAHE)", font=("Segoe UI", 10, "bold"), bg="white", fg="#34495e").pack()

        f3 = tk.Frame(img_container, bg="white")
        f3.grid(row=1, column=0, padx=20, pady=10)
        self.lbl_denoised = tk.Label(f3, bg="white")
        self.lbl_denoised.pack()
        tk.Label(f3, text="3. Gürültü Giderme (Median)", font=("Segoe UI", 10, "bold"), bg="white", fg="#34495e").pack()

        f4 = tk.Frame(img_container, bg="white")
        f4.grid(row=1, column=1, padx=20, pady=10)
        self.lbl_mask = tk.Label(f4, bg="white")
        self.lbl_mask.pack()
        tk.Label(f4, text="4. Segmentasyon (Morphology)", font=("Segoe UI", 10, "bold"), bg="white", fg="#34495e").pack()

        self.report_box = tk.Label(self.root, text="Analiz Bekleniyor...", 
                                   font=("Consolas", 12), bg="#ffffff", fg="#2c3e50", 
                                   bd=2, relief="groove", padx=20, pady=15, width=100)
        self.report_box.pack(pady=20, side="bottom")

    def analyze_random(self):
        category = self.cat_combo.get()
        cat_dir = os.path.join(self.data_path, category)
        images = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            selected_img = random.choice(images)
            self.process_and_show(os.path.join(cat_dir, selected_img))
        else:
            messagebox.showwarning("Hata", "Klasör boş!")

    def process_and_show(self, path):
        adv_features = get_advanced_features(path)
        current_category = self.cat_combo.get()
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_img[:,:,0] = clahe.apply(lab_img[:,:,0])
        enhanced_rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        
        denoised_rgb = cv2.medianBlur(img_rgb, 5)
        
        _, mask = cv2.threshold(lab_img[:,:,1], 145, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self.show_image_data(img_rgb, self.lbl_orig)
        self.show_image_data(enhanced_rgb, self.lbl_enhanced)
        self.show_image_data(denoised_rgb, self.lbl_denoised)
        self.show_image_data(final_mask, self.lbl_mask, is_mask=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_pil = Image.open(path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output, 1)
            prediction = self.classes[predicted.item()]

        _, color_mask, _ = detect_blemishes_by_color(path)
        _, _, edges = detect_texture_issues(path)
        e_score = (np.sum(edges > 0) / edges.size) * 100
        b_score = (np.sum(color_mask > 0) / color_mask.size) * 100

        report = (f"Dosya: {os.path.basename(path)}\n"
                  f"Gerçek Sınıf: {current_category} | CNN Tahmini: {prediction}\n"
                  f"Doku Düzensizliği: %{e_score:.2f} | Leke Yoğunluğu: %{b_score:.2f}\n"
                  f"Kızarıklık Skoru: {np.mean(adv_features['redness_map']):.1f} | Doku Kontrastı: {adv_features['contrast']:.1f}")
        
        text_color = "#1a73e8" if prediction == current_category else "#e74c3c"
        self.report_box.config(text=report, fg=text_color)

    def show_image_data(self, data, label, is_mask=False):
        mode = "L" if is_mask else "RGB"
        img = Image.fromarray(data, mode=mode).resize((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img)
        label.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinAnalysisApp(root)
    root.mainloop()