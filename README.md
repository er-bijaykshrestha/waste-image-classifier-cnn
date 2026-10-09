# ♻️ EcoClean: Waste Classification Using Transfer Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-FF6F00?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

> An automated machine learning pipeline leveraging Computer Vision and Transfer Learning to differentiate between organic and recyclable waste.

---

## 📖 Project Overview
Manual waste sorting is highly labor-intensive and prone to human error, often leading to the contamination of recyclable materials. This project aims to automate the waste sorting process using deep learning. By leveraging a pre-trained **VGG16** convolutional neural network (CNN) through transfer learning, the model accurately classifies images of waste into two distinct categories: **Organic (O)** and **Recyclable (R)**.

## 🎯 Aim & Objectives
The primary goal is to deploy an efficient, scalable computer vision model for real-world waste management systems. 
*   **Preprocessing:** Process image data using Keras `ImageDataGenerator` with real-time augmentation (width/height shifts, horizontal flips, and rescaling).
*   **Transfer Learning:** Implement a frozen VGG16 base model for initial feature extraction[cite: 2].
*   **Fine-Tuning:** Unfreeze later layers of the network to train on dataset-specific patterns, improving classification accuracy[cite: 2].
*   **Evaluation:** Visualize training/validation loss and accuracy curves, and output predictions on unseen test images[cite: 2].

---

## 🗄️ Dataset
This project uses a modified [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)[cite: 2]. 
The image dataset is structured into binary classes (`O` and `R`) and processed in batches of 32 at a target resolution of 150x150 pixels[cite: 2].
*   **Training Set:** 800 images[cite: 2]
*   **Validation Set:** 200 images[cite: 2]
*   **Testing Set:** 201 images[cite: 2]

---

## 🛠️ Technology Stack
| Component | Technology | Version |
| :--- | :--- | :--- |
| **Core Language** | Python | 3.12 |
| **Deep Learning** | TensorFlow / Keras | 2.17.0[cite: 2] |
| **Data Manipulation** | NumPy | 1.26.4[cite: 2] |
| **Machine Learning** | Scikit-learn | 1.5.1[cite: 2] |
| **Data Visualization** | Matplotlib | 3.9.2[cite: 2] |
| **Image Processing** | Pillow (PIL) | 12.3.0[cite: 2] |

---

## 📂 Repository Structure

```text
.
├── Proj-Classify Waste Products Using TL FT.ipynb   # Main execution notebook
├── requirements.txt                                 # Environment dependencies
├── .gitignore                                       # Git exclusion rules
├── LICENSE                                          # MIT License
└── README.md                                        # Project documentation

```
## ⚙️ Methodology & Architecture
1. Data Generators & Augmentation
Training data is augmented to prevent overfitting. The ImageDataGenerator applies a 10% width and height shift alongside horizontal flipping, while validating and testing sets are strictly rescaled by 1.0/255.0[cite: 2].

2. Feature Extraction (VGG-16)
The initial model imports the pre-trained weights from ImageNet while leaving the top classification layer out. The base layers are "frozen" (training = False), allowing the custom top layer to learn to classify the extracted features into the binary waste categories[cite: 2].

3. Fine-Tuning
After the newly added dense layers converge, specific top layers of the VGG16 base are "unfrozen." The model is then re-compiled with a very low learning rate and trained further, adapting the high-level VGG16 features specifically to the organic and recyclable waste domain[cite: 2].


### Add VGG16 transfer learning notebook for organic vs recyclable waste classification

Complete the waste classification project using transfer learning with
a pre-trained VGG16 on the Waste Classification dataset (O = organic,
R = recyclable), and compare feature extraction against fine-tuning.

## Data pipeline:
- Download and extract the reduced o-vs-r-split dataset (1200 images)
- Build ImageDataGenerators for train/val/test at 150x150, batch size 32,
  with an 80/20 train/validation split (seed 42)
- Apply augmentation (width/height shift 0.1, horizontal flip) to the
  training set only; validation and test are rescaled only
- Task 2: test_generator with shuffle=False for stable evaluation order

## Model 1 - feature extraction:
- Load VGG16 (ImageNet weights, include_top=False), flatten the output,
  and freeze all base layers
- Add head: Dense(512) -> Dropout(0.3) -> Dense(512) -> Dropout(0.3)
  -> Dense(1, sigmoid)
- Task 5: compile with binary_crossentropy, Adam (lr=1e-5), accuracy

## Model 2 - fine-tuning:
- Unfreeze from block5_conv3 onward, keep earlier layers frozen
- Compile with RMSprop (lr=1e-4)

## Training:
- 10 epochs, 5 steps per epoch
- Callbacks: exponential LR decay, EarlyStopping (val_loss, patience 4),
  and ModelCheckpoint saving the best model for each approach

## Evaluation and visualization:
- Tasks 1, 3, 4: TensorFlow version, train_generator length, model summary
- Tasks 6-8: loss and accuracy curves for both models
- Reload saved checkpoints and run both on 100 test images
  (50 O, 50 R); print classification reports
- Tasks 9-10: plot test image at index 1 with actual vs predicted label
  for each model

## Pinned dependencies: 

tensorflow 2.17.0, numpy 1.26.4, scipy 1.13.1,
scikit-learn 1.5.1, matplotlib 3.9.2.



## 👨‍💻 Author & License

Author: Bijaya Kumar Shrestha

License: This project is licensed under the MIT License. See the LICENSE file for details.

