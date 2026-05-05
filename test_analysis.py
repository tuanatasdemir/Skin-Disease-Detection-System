from src.analysis_tools import detect_blemishes_by_color, detect_texture_issues
import cv2
import matplotlib.pyplot as plt
import os

sample_path = r"data\processed\SkinDisease\one.jpeg" 

if os.path.exists(sample_path):
    original, color_mask, _ = detect_blemishes_by_color(sample_path)
    gray, lbp, edges = detect_texture_issues(sample_path)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1); plt.title("Orijinal"); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 2); plt.title("Renk Analizi (Sivilce)"); plt.imshow(color_mask, cmap='gray')
    plt.subplot(2, 2, 3); plt.title("LBP Doku Analizi"); plt.imshow(lbp, cmap='gray')
    plt.subplot(2, 2, 4); plt.title("Kenar/Kırışıklık Tespiti"); plt.imshow(edges, cmap='gray')
    
    plt.tight_layout()
    plt.show()