# ASD Conversation Classification

This project focuses on building an NLP-based classification pipeline to distinguish between conversations involving children with Autism Spectrum Disorder (ASD) and those involving neurotypical individuals. The conversations are drawn from natural, unstructured social interactions.

---

## 📁 Directory Structure

asd-classification/
├── data/
│ └── raw/ # Raw CSV files of ASD participant transcripts
├── preprocessing/
│ └── text_preprocessor.py # Class for text cleaning and normalization
├── feature_extraction/
│ └── feature_extractor.py # Class for TF-IDF and IndoBERT embedding extraction
├── model/
│ ├── trainer.py # Class for training baseline models
│ └── evaluator.py # Class for model evaluation (cross-validation)
├── notebooks/
│ └── main_experiment.ipynb # Main notebook for running experiments
├── pyproject.toml # Poetry environment configuration
└── README.md # Project documentation

## 🎯 Project Objective

- To identify linguistic characteristics in ASD children's speech using text classification techniques.
- To benchmark traditional ML models against transformer-based approaches (IndoBERT).
- To gain interpretable insights into ASD-related language patterns that could assist early research-driven screening.

---

## 🧪 Research Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Analyze sentence length, token frequency, and speaker patterns.

2. **Text Preprocessing**
   - Normalize informal terms (e.g., "nggak", "iyeee")
   - Stem using Sastrawi
   - Remove stopwords (combined from NLTK, Sastrawi, and custom set)

3. **Feature Extraction**
   - TF-IDF (traditional baseline)
   - IndoBERT embeddings (fine-tuned transformer)

4. **Modeling**
   - Baseline: Logistic Regression and SVM
   - Training with Stratified K-Fold Cross-Validation

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Outputs: Confusion Matrix and classification reports

6. **Interpretability (Planned)**
   - Use LIME or SHAP to reveal influential words/features
   - Explore attention heatmaps from BERT

7. **Early Fusion (Planned)**
   - Combine textual features with handcrafted linguistic/behavioral features

---

## 🧠 Input and Output

- **Input:** Transcription text from ASD or NON-ASD participants
- **Output:** Binary label prediction: `ASD` or `NON-ASD`
- **Main model:** IndoBERT (fine-tuned on labeled conversations)

---

## ⚠️ Limitations & Ethical Considerations

- This model is strictly for research purposes; **it is not a diagnostic tool**.
- All conversational data must be anonymized and ethically approved for use.
- The model may reflect dataset bias (e.g., regional language variants) and must be validated by domain experts before any clinical use.

---

## ⚙️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/asd-classification.git
   cd asd-classification
2. Install dependencies using Poetry:
    ```bash
    poetry install
    poetry shell

3. Launch the main experiment notebook:
    ```bash
    cd notebooks
    jupyter notebook main_experiment.ipynb


## 📂 Dataset Access

The dataset used in this project is private and not publicly available in this repository due to ethical and privacy considerations.
To request access for academic research purposes, please contact:
ahmadmufliramadhan.iclabs@umi.ac.id