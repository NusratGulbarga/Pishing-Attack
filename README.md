
# 🎯 Phishing Website Detection Using Machine Learning

## 🔍 Objective

To build a machine learning model that can classify whether a given website is a legitimate or phishing site based on its features.

---

## 🧰 Tools and Technologies Used

| Layer              | Tools & Technologies                             |
|--------------------|--------------------------------------------------|
| **Language**       | Python                                            |
| **IDE**            | Jupyter Notebook / VS Code                        |
| **ML Libraries**   | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM    |
| **Visualization**  | Matplotlib, Seaborn                               |
| **Web App (optional)** | Flask / Streamlit                            |
| **Version Control**| Git & GitHub                                      |

---

## 📜 Problem Statement

Phishing websites are designed to trick users into revealing sensitive information like usernames, passwords, or credit card numbers. The goal of this project is to use machine learning to detect whether a website is phishing or not based on extracted features.

---

## 📦 Project Scope

- Analyze and clean the phishing dataset
- Select relevant features for classification
- Train multiple ML models and evaluate their performance
- Create a simple interface to test new URLs

---

## 🧱 Modules Description

### ✅ Data Preprocessing
- Load dataset
- Handle missing values and encode categorical features

### ✅ Feature Engineering
- Use domain knowledge to extract features from URLs

### ✅ Model Training
- Train models like Logistic Regression, Random Forest, XGBoost, etc.

### ✅ Evaluation
- Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### ✅ Interface (Optional)
- A simple Flask or Streamlit app to check URLs

---

## 📊 Dataset

Use the [Phishing Websites Dataset](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector) from Kaggle.

---

## 🧠 Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost / LightGBM
- Support Vector Machine (SVM)

---

## 💡 Expected Output

- A trained model with 90%+ accuracy on test data
- Real-time prediction for new URLs
- Visual insights into important features

---

## 🗃️ Folder Structure

```bash
phishing-detection/
├── dataset/
├── models/
├── app.py
├── phishing_detection.ipynb
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/phishing-detection.git
cd phishing-detection

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebook or app
jupyter notebook phishing_detection.ipynb
# OR
python app.py
# OR
streamlit run streamlit_app.py
```

---

## 🔮 Future Enhancements

- Integrate real-time URL scanning API (VirusTotal, WHOIS)
- Add URL shortening detection
- Use NLP on page text if HTML is available

---

## 🙌 Contributors

Made with ❤️ by nusratGulbarga

---

## 📬 Contact

📧 mnusratgulbarga.2@gmail.com 
🔗 LinkedIn(https://www.linkedin.com/in/nusrat-gulbarga/) 
Portfolio (https://medium.com/@devopsdyno) 
---
