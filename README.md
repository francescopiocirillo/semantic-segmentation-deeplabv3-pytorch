# Rural Road Semantic Segmentation 🚗🛣️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Computer%20Vision-green.svg)]()

> Advanced semantic segmentation system for autonomous vehicle navigation in rural environments using DeepLabV3 and custom optimization techniques.

## 🎯 Overview

This project develops a robust **semantic segmentation pipeline** for classifying terrain elements in rural road scenarios. The system processes vehicle-mounted camera images to identify 8 distinct terrain classes essential for autonomous navigation:

- 🌅 **Sky** - Clear overhead areas
- 🛤️ **Smooth Trail** - Paved or well-maintained paths  
- 🪨 **Rough Trail** - Unpaved, rocky terrain
- 🌱 **Traversable Grass** - Safe vegetation areas
- 🌳 **High Vegetation** - Trees and tall bushes
- 🚫 **Non-traversable Low Vegetation** - Obstacle vegetation
- 💧 **Puddle** - Water hazards
- ⚠️ **Obstacle** - Physical barriers

## 🏆 Key Results

- **mIoU: 0.60+** (improved from baseline 0.51)
- **Memory Efficient**: <2.7GB VRAM training, <0.52GB inference
- **Class Imbalance Solved**: Rare classes (puddles, obstacles) improved from 0% to 56%+ IoU
- **Fast Convergence**: 8-10 epochs to optimal performance

🌍 Language Note
----------------

All **code comments and internal documentation** are written in **Italian**, as the project was developed during a group exam at the **University of Salerno (Italy)**.

Despite this, the **codebase follows international best practices**, with clear method names and class structures that make it easily understandable for global developers and recruiters.

## 🔧 Technical Architecture

### Model Pipeline
```
Input Image (RGB) → DeepLabV3 + ResNet101 → Custom ASPP → Classifier Head → 8-Class Segmentation
```

### Key Innovations
- **Custom ASPP Module**: Multi-scale context with dilations [3, 7, 3]
- **Weighted Loss Function**: Addresses severe class imbalance  
- **Selective Data Augmentation**: Targeted enhancement for rare classes
- **Auxiliary Head**: Improved gradient flow and stability
- **Smart Dropout**: 0.05 probability for noise robustness

## 📊 Performance Analysis

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| **Overall mIoU** | 0.5117 | 0.6026 | +17.8% |
| **Puddle IoU** | 0.0000 | 0.5617 | +561.7% |
| **Obstacle IoU** | ~0.20 | 0.5661 | +183% |
| **Training Time** | - | 8-10 epochs | Fast convergence |
| **Memory Usage** | - | 2.7GB train / 0.52GB inference | Efficient |

### Class-wise Performance
```
Sky:                     0.89 IoU
Smooth Trail:            0.78 IoU  
Traversable Grass:       0.72 IoU
Rough Trail:             0.65 IoU
High Vegetation:         0.61 IoU
Puddle:                  0.56 IoU ⭐ (was 0.00)
Obstacle:                0.57 IoU ⭐ (major improvement)
Non-traversable Veg:     0.19 IoU (challenging class)
```

## 📁 Project Structure

```
-
```

## 🎯 Problem-Solving Approach

### Challenge 1: Severe Class Imbalance
- **Problem**: Puddles (0% IoU), obstacles poorly detected
- **Solution**: Weighted CrossEntropy + selective oversampling  
- **Result**: Puddles improved to 56% IoU

### Challenge 2: Class Confusion (Rough vs Smooth Trail)
- **Problem**: Visually similar terrain types misclassified
- **Solution**: Texture-focused data augmentation
- **Result**: Better discrimination between trail types

### Challenge 3: Small Object Detection
- **Problem**: Missing small obstacles and puddles
- **Solution**: Custom ASPP with optimized dilation rates
- **Result**: Multi-scale context capture improved

### Challenge 4: Dataset Quality
- **Problem**: Incorrect labels in training data
- **Solution**: Manual filtering + dropout regularization
- **Result**: More robust learning despite noisy labels

## 🔬 Experimental Methodology

### Architecture Comparison
Tested multiple state-of-the-art segmentation networks:

| Architecture | Backbone | Dataset Pretraining | mIoU | Memory |
|-------------|-----------|-------------------|------|--------|
| **DeepLabV3** | **ResNet101** | **COCO** | **0.6026** | **2.7GB** |
| DeepLabV3 | ResNet50 | COCO | 0.5070 | 1.54GB |
| DeepLabV3 | MobileNetV3 | COCO | 0.5000 | <1GB |
| DeepLabV3+ | ResNet101 | Cityscapes | 0.5518 | 3.2GB |
| BiSeNetV2 | - | RUGD | 0.49 | 1.8GB |
| BiSeNetV2 | - | Rellis3D | 0.41 | 1.8GB |

### Validation Strategy
- **Stratified Split**: Ensures rare classes in both train/val
- **Memory Constraints**: <5GB training, <4GB inference
- **Metric Focus**: IoU over accuracy (more meaningful for segmentation)

## 📈 Future Improvements

- [ ] **Label Quality Enhancement**: Manual/automatic label correction
- [ ] **Domain-Specific Pretraining**: Train on Cityscapes/RUGD datasets  
- [ ] **Pseudo-Labeling**: Leverage unlabeled rural road data
- [ ] **Real-Time Optimization**: Model quantization and pruning
- [ ] **Multi-Modal Input**: Incorporate LiDAR/depth information

## 🛡️ Requirements

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **CUDA**: Compatible GPU recommended
- **RAM**: 8GB+ recommended
- **Storage**: 5GB for dataset + models

## 📋 Dependencies

Key libraries used:
```
torch>=1.9.0
torchvision>=0.10.0  
albumentations>=1.0.0
iterstrat>=0.1.2
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.21.0
Pillow>=8.0.0
tqdm>=4.60.0
```

👥 Team 6 – University of Salerno
---------------------------------

* [@francescopiocirillo](https://github.com/francescopiocirillo)
    
* [@alefaso-lucky](https://github.com/alefaso-lucky)    

* * *

📬 Contacts
-----------

✉️ Got feedback or want to contribute? Feel free to open an Issue or submit a Pull Request!

* * *

📈 SEO Tags
-----------

```
Decentralized Credential Management, Academic Credentials, Merkle Tree Credentials, Blockchain Revocation, Selective Disclosure, Privacy-Preserving Credentials, Secure Key Exchange Protocol, RSA & AES Encryption, Python Cryptography Project, Erasmus Credential Sharing, Certificate Revocation List Blockchain, Secure University Data Exchange, Project Work Algorithms and Protocols for Security, Cryptography-Based Credential System, Merkle Proof Verification, Student Credential Privacy, University of Salerno Project
```

* * *

📄 License
----------

This project is licensed under the **MIT License**, a permissive open-source license that allows anyone to use, modify, and distribute the software freely, as long as credit is given and the original license is included.

> In plain terms: **use it, build on it, just don’t blame us if something breaks**.

> ⭐ Like what you see? Consider giving the project a star!

* * *
