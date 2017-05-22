## Code for CIFAR-10 Kaggle competition

Competition page  
https://www.kaggle.com/c/cifar-10  

---
feature preprocessing: Global Contrast Normalization
#### VGG-like ConvNet
model: with 10 layers stacked + Batch Normalization + Dropout + Data Augmentation

Best performance: (< 10% of leaderboard)  
accuracy = 0.9032

#### Inception Net
model: with 15 layers (including 4 inception module) stacked + Batch Normalization + Dropout + Data Augmentation

Best performance:  
accuracy = 0.8931