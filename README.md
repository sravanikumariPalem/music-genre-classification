# Music Genre Classification using Classical Machine Learning and Deep Learning
 
## Overview
 
This project investigates automatic music genre classification across ten genres using the GTZAN dataset. Two parallel approaches are explored: a classical machine learning pipeline built on hand-crafted audio features, and a deep learning pipeline built on raw audio processed into mel spectrograms. The motivation behind running both tracks simultaneously was to understand where engineered features stand against learned representations, and whether the added complexity of a convolutional architecture justifies the cost in a dataset of this size.
 
The work sits at the intersection of signal processing, feature engineering, and modern deep learning, and is intended as a rigorous comparison rather than a push to simply maximise a single accuracy figure.
 
---
 
## Dataset
 
The GTZAN Genre Collection is the standard benchmark for music genre recognition research. It contains 1,000 audio clips of 30 seconds each, evenly distributed across 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. Each clip is sampled at 22,050 Hz in mono WAV format.
 
Two representations of the dataset are used:
 
- `features_30_sec.csv` — a pre-extracted tabular feature set covering MFCCs, chroma, spectral contrast, zero-crossing rate, tempo, and spectral centroid, among others. This feeds the classical ML pipeline.
- Raw `.wav` files — loaded and segmented programmatically to produce mel spectrograms for the CNN pipeline.
---
 
## Project Structure
 
```
music-genre-classification/
|
├── Data/
│   ├── genres_original/        # Raw WAV files organised by genre
│   └── features_30_sec.csv     # Pre-extracted tabular features
|
├── music.ipynb                 # Full pipeline notebook
└── genre_resnet_cnn.h5         # Saved ResNet CNN model
```
 
---
 
## Methodology
 
### Part 1 — Classical Machine Learning on Tabular Features
 
#### Exploratory Analysis
 
Before modelling, the feature set was examined to understand its structure and class separability. A full correlation heatmap across all numeric features identified groups of highly collinear descriptors — primarily among adjacent MFCC coefficients — which informed later decisions around scaling. Correlation of individual features with the genre label (after ordinal encoding) highlighted tempo and certain spectral features as the most discriminative.
 
A genre-level breakdown of average tempo was also visualised, confirming well-known musicological observations: metal and disco sit at the high end of the BPM range, while classical and jazz occupy the lower range. This finding, while not novel, provides a useful sanity check that the feature extraction pipeline is capturing meaningful signal.
 
Principal component analysis was applied to the standardised feature space and projected into three dimensions for visualisation. The resulting scatter plot reveals partial but imperfect separation between genres. Classical and metal tend to cluster most tightly, while genres such as rock and country show considerable overlap — a pattern that persists in the classification results and reflects genuine acoustic similarity rather than a modelling failure.
 
#### Models and Scaling Strategy
 
Five classifiers were benchmarked:
 
| Model | Scaler Applied |
|---|---|
| Random Forest (200 trees) | Standard |
| SVM with RBF kernel (C=10) | Standard |
| k-Nearest Neighbour (k=7, Euclidean) | MinMax |
| Logistic Regression (L2, C=1.0) | Standard |
| XGBoost (300 estimators, depth=6) | Standard |
 
Scaler selection per model was deliberate. SVM and Logistic Regression are sensitive to feature magnitude and benefit from zero-centred, unit-variance inputs, so StandardScaler was applied. k-NN relies on distance metrics and is particularly sensitive to features with large absolute ranges, so MinMax normalisation was used to bound all features to [0,1]. Tree-based methods (Random Forest, XGBoost) are scale-invariant by construction, but standard scaling was still applied for consistency given that XGBoost's internal gradient calculations can occasionally benefit from it.
 
All splits were stratified to preserve class balance, with 20% of the data held out for evaluation in each run.
 
#### Results
 
SVM with the RBF kernel and XGBoost were the top performers among classical models, both achieving strong macro F1-scores. The SVM's performance aligns with findings in the audio classification literature — the RBF kernel effectively handles the non-linear boundaries in a moderately high-dimensional feature space of this kind. XGBoost's competitive result is notable given it is an ensemble of shallow trees operating on the same feature space without any domain-specific tuning beyond subsample and column sampling ratios.
 
Random Forest performed well but fell marginally short of SVM and XGBoost, which is consistent with gradient boosting's general edge over bagging on structured tabular data. Logistic Regression, as expected, underperformed relative to the non-linear methods, though its result is still a meaningful baseline given the complexity of the task. k-NN ranked lowest among the five, which is expected at this scale — the curse of dimensionality begins to degrade distance-based methods even at around 50-60 features.
 
Genre-level inspection of the confusion matrices reveals that rock/country and hip-hop/reggae are the most commonly confused pairs across all models. This is acoustically interpretable: rock and country share similar tempo ranges, instrumentation, and spectral profiles, while hip-hop and reggae overlap in rhythmic structure and bass emphasis.
 
---
 
### Part 2 — Deep Learning on Raw Audio (ResNet-Inspired CNN)
 
#### Feature Representation
 
Rather than using pre-computed features, the deep learning pipeline operates directly on the audio signal. Each 30-second WAV file is segmented into overlapping 3-second clips with a 1-second hop, substantially augmenting the effective dataset size. Each segment is converted into a log-mel spectrogram with 128 mel bins, resized to 128x128 pixels, and treated as a single-channel image.
 
This representation was chosen because mel spectrograms compress the frequency axis in a way that mirrors human auditory perception, making genre-discriminative patterns — such as the dense harmonic structure of classical music or the transient-heavy percussive content of metal — visually and computationally accessible. The log compression reduces dynamic range and makes the representation more stationary across different recording levels.
 
#### Architecture
 
The CNN architecture is a lightweight ResNet-inspired network built using the Keras Functional API. The key modification over a plain sequential CNN is the introduction of residual (skip-connection) blocks throughout the network. Each residual block adds the block's input directly to its output, bypassing the convolution layers. Where the spatial resolution or channel count changes, a 1x1 projection convolution aligns the shortcut dimensions before the addition.
 
The architecture follows this structure:
 
```
Input (128x128x1)
  -> Stem: Conv2D(32, 7x7, stride=2) + BN + ReLU + MaxPool(3x3, stride=2)
  -> Stage 1: ResBlock(32) x2                         [same resolution]
  -> Stage 2: ResBlock(64, downsample) + ResBlock(64)  [spatial /2]
  -> Stage 3: ResBlock(128, downsample) + ResBlock(128) [spatial /2]
  -> GlobalAveragePooling2D
  -> Dense(256, ReLU) + Dropout(0.45)
  -> Dense(10, Softmax)
```
 
The motivation for using residual connections rather than a plain deep CNN is gradient stability. In a deep sequential network, the gradient signal attenuates as it propagates backwards through many layers. Residual connections provide a direct path for gradients to flow to earlier layers, allowing the network to train more reliably at greater depth without resorting to careful initialisation schemes.
 
GlobalAveragePooling replaces the traditional Flatten-Dense head. This reduces the parameter count considerably, acts as a form of regularisation, and makes the model more robust to small variations in the input spectrogram shape.
 
#### Training
 
The model was trained with the Adam optimiser at an initial learning rate of 5e-4, using categorical cross-entropy loss. Two callbacks were employed:
 
- `EarlyStopping` monitored validation accuracy with a patience of 7 epochs, restoring the best weights on termination.
- `ReduceLROnPlateau` halved the learning rate when validation loss plateaued for 4 consecutive epochs, down to a floor of 1e-6.
A 75/25 train-test split was used, stratified by class. Batch size was set to 32.
 
#### Results
 
The CNN pipeline operated on significantly more samples than the tabular pipeline due to audio segmentation, which partially offsets the limited size of the original GTZAN dataset. Learning curves showed consistent improvement in both training and validation accuracy across epochs, with the learning rate decay visibly improving convergence stability in the later stages of training.
 
The ResNet CNN's confusion matrix shows clearer separation between classical and most other genres compared to the classical ML models, which is expected — the spectrogram representation captures the sparse, harmonic structure of classical music in a way that aggregate statistical features partially lose. Metal similarly separates well, due to its distinct high-energy, full-spectrum signature.
 
The persistent confusions between rock/country and reggae/hip-hop remain present in the CNN results as well, suggesting these overlaps are a property of the data rather than a limitation of any particular model family.
 
---
 
## Technical Stack
 
| Component | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Classical ML | scikit-learn |
| Gradient boosting | XGBoost |
| Audio processing | librosa |
| Deep learning | TensorFlow / Keras |
| Dimensionality reduction | scikit-learn PCA |
 
---
 
## Reproducing the Results
 
1. Clone the repository and install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost librosa tensorflow
```
 
2. Download the GTZAN dataset and place the `genres_original` folder and `features_30_sec.csv` under `Data/`.
3. Update the path constants `DATASET_ROOT` and `AUDIO_DIR` in the notebook to point to your local `Data/` directory.
4. Run all cells in `music.ipynb` sequentially. The classical ML benchmark runs independently of the CNN pipeline, so sections 3-8 can be executed without librosa or TensorFlow if only the tabular results are needed.
The trained CNN is saved to `genre_resnet_cnn.h5` at the end of the notebook for later inference or fine-tuning.
 
---
 
## Discussion
 
The central question motivating the dual-track design was whether raw audio modelling via CNNs materially improves over well-engineered tabular features on a dataset of this size. The results suggest the answer is nuanced.
 
The classical ML models — particularly SVM and XGBoost — achieve competitive accuracy with significantly lower computational cost and no dependency on GPU hardware. On a 1,000-clip dataset this matters: the CNN's additional performance gains come from audio segmentation that multiplies the effective training set, not purely from architectural superiority.
 
If the dataset were larger, the gap would likely widen in favour of the CNN. The tabular pipeline is bounded by the information captured in aggregate statistics; spectrograms preserve temporal structure that features like mean MFCC coefficients discard. For genres where temporal evolution is a defining characteristic — progressive rock, jazz improvisation, classical development — this would become increasingly important at scale.
 
The residual architecture was chosen specifically to allow the network to go deeper without degrading training dynamics. On a dataset this size the depth is arguably unnecessary, but the design decision reflects a preference for building with architectures that generalise cleanly to larger problems, rather than overfitting the solution to the current data regime.
 
---
 
## Limitations
 
The GTZAN dataset has well-documented issues. Several clips contain duplicate recordings, and the genre boundaries were assigned by a single annotator, introducing label subjectivity particularly for stylistically ambiguous tracks. These factors mean that absolute accuracy figures should be interpreted carefully rather than compared directly to results on other datasets.
 
The tabular features were computed externally and ingested as a CSV. A more rigorous pipeline would extract features directly from the WAV files to ensure consistency with the CNN preprocessing.
 
---
