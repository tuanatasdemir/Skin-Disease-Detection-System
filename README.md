# Cilt Analizi ve Teşhis Sistemi 

Bu proje, **Tekirdağ Namık Kemal Üniversitesi** Görüntü İşleme dersi kapsamında geliştirilmiştir. Sistem, derin öğrenme ve geleneksel görüntü işleme tekniklerini birleştirerek cilt hastalıklarını analiz eder.

## Kullanılan Teknolojiler
* **Python** & **OpenCV** (Görüntü İşleme)
* **PyTorch** (Derin Öğrenme / CNN)
* **Tkinter** (Kullanıcı Arayüzü)
* **Scikit-Image** (GLCM Doku Analizi)

## Uygulanan Teknikler
1. **CLAHE:** Yerel kontrast iyileştirme.
2. **Median Filter:** Gürültü giderme.
3. **Morphology:** Morfolojik açınım ile segmentasyon temizliği.
4. **SkinCNN:** 6 farklı kategoriyi tanıyan özel evrişimli sinir ağı.

## Kurulum ve Çalıştırma
Projeyi yerel bilgisayarınızda çalıştırmak için şu adımları izleyin:

1. Kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Modeli eğitin:
   ```bash
   python train_pytorch.py
   ```
3. Uygulamayı başlatın:
   ```bash
   python main_gui.py
   ```

## Analiz Çıktıları
Uygulama; Leke Yoğunluğu (%), Doku Düzensizliği (%) ve Kızarıklık Skoru gibi nicel veriler üretmektedir.

