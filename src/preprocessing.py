import kagglehub
import shutil
import os
import cv2
import numpy as np

download_path = kagglehub.dataset_download("pacificrm/skindiseasedataset")

target_path = "data/raw/skindiseasedataset"

if not os.path.exists(target_path):
    print(f"Veriler taşınıyor: {target_path}")
    shutil.move(download_path, target_path)
    print("Taşıma işlemi tamamlandı.")
else:
    print("Veri seti zaten data/raw klasöründe mevcut.")

categories = os.listdir(target_path)
print(f"Tespit edilen cilt kategorileri: {categories}")

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    
    return img_normalized

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def process_all_data(raw_dir, processed_dir):
    actual_raw_path = os.path.join(raw_dir, "SkinDisease")
    if not os.path.exists(actual_raw_path):
        actual_raw_path = raw_dir

    categories = os.listdir(actual_raw_path)
    
    for cat in categories:
        cat_path = os.path.join(actual_raw_path, cat)
        
        if not os.path.isdir(cat_path):
            continue
            
        save_path = os.path.join(processed_dir, cat)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"İşleniyor: {cat}...")
        
        for img_name in os.listdir(cat_path):
            img_full_path = os.path.join(cat_path, img_name)

            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                processed_img = preprocess_image(img_full_path)
                
                if processed_img is not None:
                    final_img = (processed_img * 255).astype(np.uint8)
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_path, img_name), final_img)

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/skindiseasedataset"
    PROCESSED_DATA_PATH = "data/processed"

    print("İşlem başlatılıyor...")
    process_all_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print(f"İşlem tamamlandı. Çıktıları kontrol et: {PROCESSED_DATA_PATH}")
