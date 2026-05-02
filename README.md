# 🛡️ Credit Card Fraud Detection Using ML&DL — Graduation Project🎓

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**🎓 Final Year Graduation Project — TM471-II**

*A complete end-to-end Machine Learning system: from raw imbalanced data to a deployed REST API web application capable of detecting fraudulent credit card transactions in real time.*

> **Supervisor:** DR. Mahmoud Ata-Allah &nbsp;|&nbsp; **Student:** Muhammed Abd Elmoneim Nabawy Fathallah &nbsp;|&nbsp; **ID:** 2051710891

</div>

---

## 📌 Executive Summary

Credit card fraud causes **billions of dollars in losses** every year. This project tackles the problem end-to-end — from raw, highly imbalanced data all the way to a live prediction API — using multiple machine learning algorithms, advanced sampling techniques, and a neural network to find the most reliable fraud detector.

### Key Highlights
- Trained and compared **4 classical ML models** + a **Neural Network** (Keras/TensorFlow)
- Solved the critical **class imbalance problem** using both Random UnderSampling and **SMOTE** oversampling
- Built and deployed a **Flask REST API** with a real-time prediction web interface
- Worked on a real-world Kaggle dataset: **284,807 transactions**, only **0.17% fraudulent**
- Evaluated using ROC-AUC, Precision, Recall, F1-Score — not just accuracy (which is misleading on imbalanced data)

---

## 📊 Dataset

| Property | Value |
|---|---|
| 📂 Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/muhammedabdelmoneim/european-cardholders-in2013-284-000-transactions) |
| 📝 Total Transactions | **284,807** |
| 🚨 Fraudulent Transactions | **492 (0.17%)** |
| ✅ Legitimate Transactions | **284,315 (99.83%)** |
| 🔢 Features | **31** (Time, Amount + V1–V28 via PCA) |
| ⚖️ Class Imbalance | Extreme — required special handling |

> The dataset contains transactions by European cardholders in September 2013. Features V1–V28 are the result of PCA transformation to protect user privacy. Only `Time` and `Amount` are in their original form.

---

## 🧠 ML Pipeline Overview

```
Raw Data (284,807 txns)
        │
        ▼
  EDA & Visualization
  (distributions, correlation heatmaps, boxplots)
        │
        ▼
  Feature Scaling
  (RobustScaler on Amount & Time — less sensitive to outliers)
        │
        ▼
  Handling Class Imbalance
  ┌──────────────────┐   ┌────────────────────┐
  │ Random Under-    │   │ SMOTE Over-        │
  │ Sampling         │   │ Sampling           │
  └──────────────────┘   └────────────────────┘
        │
        ▼
  Model Training & Cross-Validation (StratifiedKFold)
  ┌────────────────────────────────────────────┐
  │  Logistic Regression  │  KNN               │
  │  SVM (SVC)            │  Decision Tree     │
  │  Neural Network (Keras/TensorFlow)         │
  └────────────────────────────────────────────┘
        │
        ▼
  Evaluation (ROC-AUC, Precision, Recall, F1, Confusion Matrix)
        │
        ▼
  Best Model Saved → model.pkl
        │
        ▼
  Flask REST API + Web UI Deployment
```

---

## 🤖 Models & Results

### Classical ML — Trained on UnderSampled & SMOTE Data

| Model | Sampling Strategy | Notes |
|---|---|---|
| **Logistic Regression** | SMOTE (best config) | ⭐ Best overall — chosen for deployment |
| **Support Vector Classifier (SVC)** | UnderSampling | Strong recall on fraud class |
| **K-Nearest Neighbors (KNN)** | UnderSampling | Decent but slower inference |
| **Decision Tree** | UnderSampling | Fast but prone to overfitting |

### Neural Network (Keras / TensorFlow)

```
Architecture:
Input Layer (n_features nodes, ReLU)
    → Hidden Layer (32 nodes, ReLU)
    → Output Layer (2 nodes, Softmax)

Optimizer : Adam (lr=0.001)
Loss      : Sparse Categorical Crossentropy
Epochs    : 20  |  Batch Size: 25–300
```

Tested on both UnderSampled and SMOTE-oversampled subsets to compare generalization on the original test set.

### Why Logistic Regression Was Selected for Deployment

The best model was chosen not by accuracy alone (misleading on imbalanced data), but by:
- **ROC-AUC Score** — ability to separate fraud from non-fraud
- **Recall on fraud class** — minimizing missed fraudulent transactions
- **Precision-Recall tradeoff** — balancing false alarms vs. missed detections

Logistic Regression + **SMOTE** produced the best balance of all three.

---

## 🌐 Web Application (Flask REST API)

A full REST API with an HTML/CSS frontend that lets anyone interact with the model in real time — no code needed.

### How It Works

```
User enters 30 feature values
        │
        ▼
Flask receives POST request → parses features → reshapes to numpy array
        │
        ▼
clf.predict() → 0 (Legitimate) or 1 (Fraudulent)
        │
        ▼
Result rendered on result.html
```

### Running the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py

# 4. Open your browser at http://localhost:5000
```

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── 📄 README.md
├── 📄 .gitignore
│
├── 📓 creditcard_notebook.ipynb       ← Full ML pipeline: EDA → Training → Evaluation
│
├── app.py                             ← Flask REST API backend
├── model.pkl                          ← Saved trained model (Logistic Regression)
├── saved_model.pkl                    ← Backup model
│
├── templates/
│   ├── home.html                      ← Prediction input form
│   └── result.html                    ← Prediction result page
│
├── static/
│   └── style.css                      ← Frontend styling
│
├── Dataset.csv                        ← Credit card transactions dataset
│
└── docs/
    └── Graduation_Project_Final_report.docx
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (LR, SVM, KNN, Decision Tree) |
| **Deep Learning** | TensorFlow / Keras |
| **Imbalance Handling** | imbalanced-learn (SMOTE, NearMiss) |
| **Dimensionality Viz** | t-SNE, PCA, TruncatedSVD |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3 |
| **Model Persistence** | Pickle |
| **Development** | Jupyter Notebook, PyCharm |

---

## 🔍 Key Challenges & How They Were Solved

| Challenge | Solution |
|---|---|
| **Extreme class imbalance** (0.17% fraud) | Applied Random UnderSampling AND SMOTE — compared both |
| **Misleading accuracy metric** | Used ROC-AUC, Precision, Recall, F1-Score instead |
| **Feature privacy** (PCA-anonymized data) | Worked with V1–V28 as-is; scaled Time & Amount with RobustScaler |
| **Model selection** | Trained 5 models, StratifiedKFold cross-validation, selected best |
| **Deployment** | Serialized model with Pickle, served via Flask REST API |

---

## 📐 System Design (UML)

Full UML documentation was produced before implementation:

| Diagram | What It Shows |
|---|---|
| **Use Case** | Cardholder, Merchant, Bank interacting with the fraud detection system |
| **Activity** | Transaction flow: initiation → verification → fraud check → outcome |
| **Class** | Cardholder, Transaction, Merchant, Bank, FraudDetectionSystem relationships |
| **Sequence** | Message exchange between all components over time |

---

## ⚖️ Ethical, Legal & Social Considerations

- **Privacy** — All features are PCA-anonymized; no personally identifiable information is stored or exposed
- **Legal compliance** — Design respects data privacy regulations and financial industry standards
- **Fairness** — Model evaluated to avoid systematic false positives that unfairly block legitimate users
- **Social impact** — Effective fraud detection protects consumers and builds trust in digital payments

---

## 📚 Related Work & What Makes This Project Different

| Project | Approach | Web App |
|---|---|---|
| [Laurent Veyssier](https://github.com/LaurentVeyssier/Credit-Card-fraud-detection-using-Machine-Learning) | Random Forest, Neural Net, SMOTE | ❌ |
| [Vipul Jain (IEEE)](https://ieeexplore.ieee.org/document/9915901) | Random Forest, AdaBoost, Streamlit | ✅ Streamlit |
| [Vejalla et al. (IEEE)](https://ieeexplore.ieee.org/document/10136040) | KNN, SVM, Naive Bayes, Random Forest | ❌ |
| **This Project** | LR, KNN, SVM, DT + Neural Network, SMOTE | ✅ **Flask REST API** |

**Differentiators:**
- Full pipeline from raw data to a **live, interactive REST API**
- **5 models** compared across **2 sampling strategies** in a single notebook
- Emphasis on **correct evaluation metrics** for imbalanced classification (not just accuracy)
- Complete software engineering: UML diagrams, SDLC, functional & non-functional requirements

---

## 🚀 Future Work

- [ ] Real-time streaming for live transaction monitoring
- [ ] Experiment with XGBoost, LightGBM, and ensemble stacking
- [ ] Cloud deployment (AWS / Azure / Google Cloud)
- [ ] Database integration to log predictions for continuous retraining
- [ ] SHAP values for model explainability and interpretability
- [ ] Admin dashboard for fraud analysts

---

## 👤 Author

**Muhammed Abd Elmoneim Nabawy Fathallah**  
Student ID: 2051710891  
Final Year Project — TM471-II  
Supervisor: DR. Mahmoud Ata-Allah

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ☕ and a lot of confusion matrices</sub>
</div>
