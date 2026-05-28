# 🎵 Music Genre Classification
 
Classifying music genres using classical machine learning and deep learning on the GTZAN dataset. Compares feature-engineered ML models against a CNN trained on Mel spectrograms.
 
---
 
## 📋 Overview
 
This project explores two approaches to automatic music genre classification:
 
1. **Classical ML** — hand-crafted audio features (MFCCs, chroma, tempo, zero-crossing rate) fed into Random Forest, SVM, k-NN, and Logistic Regression classifiers
2. **Deep Learning (CNN)** — Mel spectrograms generated from 3-second audio segments used to train a custom Convolutional Neural Network
Key finding: k-NN achieved the highest accuracy at **89%**, outperforming the CNN (83%), demonstrating that well-engineered features can match deep learning on constrained audio datasets.
 
---
 
## 📊 Results
 
| Model               | Accuracy |
|---------------------|----------|
| k-NN                | **89%**  |
| Random Forest       | 87%      |
| SVM (RBF kernel)    | 86%      |
| CNN (Mel spectrogram) | 83%   |
| Logistic Regression | 72%      |
 
---
 
## 🗂️ Dataset
 
**GTZAN Genre Collection**
- 1,000 audio tracks (30 seconds each)
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Pre-extracted features available as `features_3_sec.csv`
Download from [Kaggle – GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
 
Place the dataset in:
```
Data/
├── genres_original/
│   ├── blues/
│   ├── classical/
│   └── ...
└── features_3_sec.csv
```
 
---
 
## 🛠️ Tech Stack
 
- **Language:** Python 3.x
- **ML:** scikit-learn (Random Forest, SVM, k-NN, Logistic Regression)
- **Deep Learning:** TensorFlow / Keras
- **Audio Processing:** librosa
- **Data & Visualisation:** pandas, NumPy, matplotlib, seaborn
---
 
## 🚀 Getting Started
 
### 1. Clone the repository
```bash
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification
```
 
### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow librosa
```
 
### 3. Run the notebook
```bash
jupyter notebook mlmodel.ipynb
```
 
---
 
## 🧠 Methodology
 
### Classical ML Pipeline
- Features extracted from `features_3_sec.csv`: MFCCs (mean & variance), chroma, tempo, spectral features, zero-crossing rate
- Scaler selection per model: StandardScaler for RF/SVM/LR, MinMaxScaler for k-NN
- Stratified 80/20 train-test split with `random_state=68`
- Evaluated with accuracy, macro F1-score, macro recall, and confusion matrix
### CNN Pipeline
- Raw `.wav` files loaded with `librosa` at 22,050 Hz
- Each track segmented into **3-second clips** with 1-second overlap to increase data variability and reduce overfitting
- Mel spectrograms generated at 128 mel bands, resized to 128×128 and saved as single-channel images
- CNN architecture: 3× Conv2D → BatchNorm → MaxPool → Dropout blocks, followed by Dense(256) + Dropout(0.5) + Softmax output
- Trained for 30 epochs with Adam (lr=0.0001), categorical cross-entropy loss
- Trained model saved as `genre_cnn_model.h5`
### Dimensionality Reduction (EDA)
- PCA applied to standardised features to visualise genre separability in 3D
- Feature correlation heatmap generated to identify redundant features
- Per-genre tempo distributions plotted to validate feature relevance
---
 
## 📁 Project Structure
 
```
music-genre-classification/
├── mlmodel.ipynb          # Main notebook (ML + CNN)
├── genre_cnn_model.h5     # Saved CNN model (generated after training)
├── Data/
│   ├── features_3_sec.csv
│   └── genres_original/
└── README.md
```
 
---
 
## 💡 Key Insights
 
- **MFCCs and chroma** were the most predictive features; tempo and zero-crossing rate showed weaker correlation with genre labels
- **k-NN with MinMax scaling** outperformed all other models, suggesting genre boundaries are locally clustered in feature space
- **CNN performance (83%)** confirms deep learning viability on image-like audio representations, though it requires significantly more compute than classical ML
- 3-second segmentation improved CNN generalisation by increasing effective sample size per track
---
 
## 📄 License
 
MIT License — free to use and adapt with attribution.
