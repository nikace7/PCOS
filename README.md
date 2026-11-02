# PCOS Ultrasound Diagnostic Classifier

A deep learning Convolutional Neural Network (CNN) pipeline developed with **TensorFlow** and **Keras** for automated screening and binary classification of **Polycystic Ovary Syndrome (PCOS)** from pelvic ultrasound imaging.

<br/>

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-005222?style=flat-square&logo=tensorflow&logoColor=f7f5f1)](https://www.tensorflow.org/)
&nbsp;
[![Keras](https://img.shields.io/badge/Keras-CNN_Architecture-005222?style=flat-square&logo=keras&logoColor=f7f5f1)](https://keras.io/)
&nbsp;
[![Biomedical](https://img.shields.io/badge/Biomedical-Ultrasound_Diagnosis-007136?style=flat-square)](https://github.com/nikace7/PCOS)
&nbsp;
[![Python](https://img.shields.io/badge/Python-3.10%2B-15181e?style=flat-square&logo=python&logoColor=f7f5f1)](https://www.python.org/)
&nbsp;
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-15181e?style=flat-square&logo=jupyter&logoColor=f7f5f1)](https://jupyter.org/)
&nbsp;
[![Image Processing](https://img.shields.io/badge/Image_Processing-PIL_·_Pillow-15181e?style=flat-square)](https://pillow.readthedocs.io/)

---

### Overview

Polycystic Ovary Syndrome (PCOS) is an endocrine disorder affecting an estimated 8–13% of women of reproductive age. Traditional diagnosis relies heavily on transvaginal or transabdominal pelvic ultrasound to identify distinctive morphological markers—namely peripheral micro-follicles ("string of pearls" sign) and stromal hyperplasia. 

Manual ultrasound assessment can be time-consuming and subjective. This project implements an automated computer vision pipeline that processes raw ultrasound scans, applies robust medical image augmentation, and classifies images into **PCOS-positive** or **Normal (Control)** cohorts.

---

### Deep Learning Pipeline & Model Architecture

```
+-------------------------------------------------------------------------------+
|                       Ultrasound Scan (128 x 128 x 3)                         |
+---------------------------------------+---------------------------------------+
                                        |
                   Safe Data Ingestion  |  Random Rotation (±15°)
                   Rescaling (1/255)    |  Width/Height Shifts (10%)
                                        |  Shear / Zoom (10%)
                                        v
+-------------------------------------------------------------------------------+
|                        Convolutional Feature Extractor                        |
|                                                                               |
|   Block 1: Conv2D (32 filters, 3x3, ReLU)  -->  MaxPooling2D (2x2)            |
|   Block 2: Conv2D (64 filters, 3x3, ReLU)  -->  MaxPooling2D (2x2)            |
|   Block 3: Conv2D (128 filters, 3x3, ReLU) -->  MaxPooling2D (2x2)            |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                            Classification Head                                |
|                                                                               |
|   Flatten()  -->  Dense (128 units, ReLU)  -->  Dropout (p=0.5)               |
|                                            -->  Dense (1 unit, Sigmoid)       |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|            Output: [0: Normal]  or  [1: PCOS Confirmed] (Confidence)          |
+-------------------------------------------------------------------------------+
```

#### 1. Fault-Tolerant Medical Ingestion (`SafeImageDataGenerator`)
Medical imaging datasets frequently contain corrupted headers, partial writes, or unsupported bit depths that cause standard Keras generators to crash mid-epoch. This project defines an overridden `SafeImageDataGenerator` that wraps PIL ingestion with error guards:
```python
class SafeImageDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        # Gracefully skips corrupted images (UnidentifiedImageError, OSError)
        # without halting the training loop or introducing null tensor offsets.
```

#### 2. Domain-Specific Data Augmentation
To counter the limited sample size of clinical imaging and prevent the model from memorizing ultrasound machine watermarks or gain settings, data is augmented during training:
- Pixel Normalization: `rescale=1./255`
- Rotation Range: `15°`
- Spatial Translations: `width_shift_range=0.1`, `height_shift_range=0.1`
- Geometric Distortions: `shear_range=0.1`, `zoom_range=0.1`
- Mirroring: `horizontal_flip=True`

#### 3. Regularization & Training Callbacks
To prevent overfitting on medical features, training is protected by three complementary callbacks:
- **`ModelCheckpoint`:** Monitors `val_loss` and serializes only the optimal weights to `best_model.keras`.
- **`EarlyStopping`:** Halts training if validation loss fails to improve across 10 consecutive epochs (`patience=10`).
- **`ReduceLROnPlateau`:** Dynamically scales down the Adam learning rate by a factor of 5 (`factor=0.2`, `patience=5`, `min_lr=1e-5`) when approaching loss plateaus.

---

### Project Structure

```
PCOS/
├── SASPCOS.py             # Full end-to-end training pipeline & CNN definition
├── test.py                # Standalone inference utility for single-scan evaluation
├── Untitled.ipynb         # Interactive Jupyter research notebook (EDA & evaluation)
├── best_model.keras       # Serialized production model weights (HDF5 / Keras format)
├── sample_images/         # Ultrasound test samples (.jpg, .png)
└── README.md
```

---

### Getting Started

#### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** with CUDA support recommended for model training (TensorFlow automatically falls back to CPU if unavailable).

#### 1. Clone & Set Up Environment
```bash
git clone https://github.com/nikace7/PCOS.git
cd PCOS

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install tensorflow numpy pillow jupyter matplotlib
```

#### 2. Running Inference on an Ultrasound Scan
To run a prediction against an ultrasound image:
```bash
python test.py
```

Under the hood, `test.py` loads `best_model.keras`, resizes the target image to `(128, 128)`, normalizes the pixel matrix, and outputs the diagnostic decision:
```python
from test import predict_image
import tensorflow as tf

model = tf.keras.models.load_model('best_model.keras')
label, confidence = predict_image('path_to_ultrasound.jpg', model)

print(f"Prediction: {'PCOS Detected' if label == 1 else 'Normal'}")
print(f"Confidence Score: {confidence[0][0]:.4f}")
```

#### 3. Retraining the CNN
To train the model on a new or expanded dataset:
1. Organize your dataset directory structure:
   ```
   dataset/
   ├── train/
   │   ├── pcos/
   │   └── normal/
   └── test/
       ├── pcos/
       └── normal/
   ```
2. Update `train_dir` and `test_dir` in `SASPCOS.py`.
3. Run the training script:
   ```bash
   python SASPCOS.py
   ```

---

### Clinical Disclaimer

> **Important:** This repository and its accompanying weights are intended strictly for academic research and educational exploration in biomedical computer vision. They are **not** certified as medical diagnostic devices and should not be used for clinical decision-making or diagnosis without clinical validation and regulatory oversight.

---

### Author

**Nikesh Khatri**  
Software Engineer — Kathmandu, Nepal  
- Website: [nikeshkhatri.com.np](https://nikeshkhatri.com.np)  
- GitHub: [@nikace7](https://github.com/nikace7)  
- LinkedIn: [Nikesh Khatri](https://www.linkedin.com/in/nikesh-khatri-2b04582a9/)  
- Email: [khatrinikace@gmail.com](mailto:khatrinikace@gmail.com)
