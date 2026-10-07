# 🩺 AI-Based Liver Disease Prediction

An end-to-end machine learning application for estimating the likelihood of liver disease using the **Indian Liver Patient Dataset (ILPD)**. The project combines **data preprocessing, feature scaling, PCA dimensionality reduction, six machine learning classifiers, model evaluation, Explainable AI (XAI), an AI Liver Health Assistant, and an interactive Streamlit dashboard**.

## 🚀 Project Pipeline

```text
Indian Liver Patient Dataset (ILPD)
                ↓
       Data Preprocessing
                ↓
        Feature Scaling
                ↓
              PCA
        (95% Variance)
                ↓
       6 ML Classifiers
                ↓
        Model Evaluation
                ↓
      Best Model Selection
         (Highest F1)
                ↓
           Prediction
                ↓
    Explainable AI (XAI)
                ↓
   AI Liver Health Assistant
                ↓
      Streamlit Dashboard
                ↓
             Report
```

## ✨ Features

* Indian Liver Patient Dataset (ILPD) based prediction
* Data cleaning and preprocessing
* Missing-value handling
* Feature scaling using `StandardScaler`
* PCA dimensionality reduction
* Comparison of six machine learning classifiers
* Evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC
* Confusion Matrix
* Automatic best-model selection based on F1-score
* Explainable AI using permutation importance
* Patient-level feature sensitivity analysis
* Reference-range checking for clinical input values
* AI Liver Health Assistant
* Interactive Streamlit dashboard
* Automatic prediction report generation
* Medical safety disclaimer

## 🤖 Machine Learning Models

The project evaluates six classifiers:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes

The model with the **highest F1-score on the held-out test set** is selected as the final model.

## 📊 Model Results

| Model               |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression |     71.9% |     90.2% |     67.9% |     77.5% |     80.4% |
| Decision Tree       |     64.0% |     83.3% |     61.7% |     70.9% |     68.6% |
| Random Forest       |     71.1% |     75.0% |     88.9% |     81.4% |     72.2% |
| SVM                 |     71.9% |     94.5% |     64.2% |     76.5% |     82.0% |
| **KNN**             | **71.1%** | **74.5%** | **90.1%** | **81.6%** | **68.1%** |
| Naive Bayes         |     66.7% |     86.4% |     63.0% |     72.9% |     77.7% |

### 🏆 Best Model: KNN

KNN is selected as the final model because it achieved the highest **F1-score of 81.6%**.

**KNN Results:**

* Accuracy: **71.1%**
* Precision: **74.5%**
* Recall: **90.1%**
* F1-score: **81.6%**
* ROC-AUC: **68.1%**

> KNN is selected based on F1-score and is not the best-performing model for every individual metric.

## 🔬 PCA

**Principal Component Analysis (PCA)** is used to reduce the dimensionality of the original clinical features while retaining most of the information.

* Original features: **10**
* PCA components retained: **7**
* Target variance: **95%**
* PCA is fitted only on the training data through the Scikit-learn pipeline to avoid data leakage.

```text
10 Original Features
        ↓
       PCA
        ↓
7 Principal Components
        ↓
  ML Classifier
```

## 🔍 Explainable AI (XAI)

The project provides explanations using the original clinical features instead of displaying only PCA components.

### Global Explanation

**Permutation importance** is used to estimate how much model performance changes when individual features are shuffled. This helps identify which clinical features are relatively more influential to the trained model.

### Patient-Level Explanation

The dashboard provides a patient-level feature sensitivity analysis by changing individual input features and observing changes in the model's estimated probability.

> These explanations describe model behavior and feature sensitivity. They should not be interpreted as causal medical explanations.

## 🩺 AI Liver Health Assistant

The application includes an AI-based assistant that explains model output in simple language.

The assistant can provide information about:

* Model prediction
* Estimated probability
* Important model features
* Reference-range abnormalities
* Liver-related terminology
* Model reliability and limitations
* General next-step guidance

The project includes a **built-in rule-based assistant**, so it can work without an external API.

An optional LLM-based assistant can be enabled using an Anthropic API key.

## 🖥️ Streamlit Dashboard

The application contains five main sections:

### 1. Input & Prediction

Enter patient parameters and obtain the model's estimated probability.

### 2. Explanation (XAI)

View:

* Patient-level feature sensitivity
* Global permutation importance
* Feature contribution information

### 3. AI Liver Health Assistant

Ask questions about:

* The prediction
* Model behavior
* Clinical feature terminology
* Reference ranges
* Model limitations

### 4. Model Comparison

Compare all six classifiers using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* PCA information

### 5. Report

Generate a structured report containing:

* Patient inputs
* Prediction
* Estimated probability
* Feature explanation
* Reference-range checks
* AI summary
* Recommendation
* Medical disclaimer

## 📁 Project Structure

```text
liver_ai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── indian_liver_patient.csv
│   └── make_demo_data.py
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── train.py
│   ├── explain.py
│   └── assistant.py
│
├── models/
│   ├── artifacts.joblib
│   └── metrics.json
│
├── smoke_test.py
└── apptest.py
```

## ⚙️ Installation

Clone or download the repository and open the project directory.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Dataset

Place the real **Indian Liver Patient Dataset (ILPD)** at:

```text
data/indian_liver_patient.csv
```

The project also contains:

```text
data/make_demo_data.py
```

which can generate synthetic demonstration data if the real dataset is unavailable.

> **Important:** Synthetic demo data must not be used for reporting final model performance. The final project should be trained and evaluated using the real ILPD dataset.

## 🏋️ Train the Models

After placing the real dataset in the correct location, run:

```bash
python -m src.train
```

This will:

1. Load and clean the dataset
2. Separate features and target
3. Split the data into training and testing sets
4. Apply preprocessing
5. Scale the features
6. Apply PCA
7. Train six classifiers
8. Calculate evaluation metrics
9. Select the best model using F1-score
10. Save the trained model artifacts
11. Save the evaluation results

Generated files:

```text
models/artifacts.joblib
models/metrics.json
```

## ▶️ Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## 🤖 Optional AI Assistant

The project works with the built-in rule-based assistant by default.

An optional Anthropic-powered assistant can be enabled using:

```bash
ANTHROPIC_API_KEY=your_api_key
```

The application will fall back to the built-in assistant if the API is unavailable.

## 🧪 Testing

Run the smoke test:

```bash
python smoke_test.py
```

Run the application test:

```bash
python apptest.py
```

## 🛠️ Technologies Used

* **Programming:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Dimensionality Reduction:** PCA
* **Visualization:** Matplotlib
* **Web Dashboard:** Streamlit
* **Model Persistence:** Joblib
* **Explainable AI:** Permutation Importance
* **AI Assistant:** Rule-based assistant, optional Anthropic LLM integration

## 🎯 Project Objective

The primary objective of this project is to demonstrate how machine learning can be combined with dimensionality reduction and Explainable AI to build an interactive system for liver disease risk estimation.

The project focuses on:

```text
Machine Learning
       +
PCA Dimensionality Reduction
       +
Model Comparison
       +
Explainable AI
       +
AI Assistant
       +
Interactive Dashboard
```

## ⚠️ Medical Disclaimer

This project is developed for **educational and academic purposes**.

The prediction is a machine-learning screening estimate and **is not a medical diagnosis**. The model may produce incorrect predictions and should not be used to make medical decisions.

Always consult a qualified healthcare professional for interpretation of medical test results, diagnosis, treatment, and next steps.

## 👨‍💻 Project

**AI-Based Liver Disease Prediction**

**Technologies:** Python, Pandas, NumPy, Scikit-learn, PCA, Streamlit, Explainable AI

**Dataset:** Indian Liver Patient Dataset (ILPD)
