## 🧠 Brainbox -AI

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/jayanth-reddy31/Chunkwise/main/chunkwise_UI_1.png" width="600"/></td>
    <td><img src="https://raw.githubusercontent.com/jayanth-reddy31/Chunkwise/main/chunkwise_UI_2.png" width="600"/></td>
  </tr>
</table>

# All-in-One ML & DL Streamlit Web App

This Streamlit web application integrates a collection of **machine learning** and **deep learning** models for a variety of classification and regression tasks. It provides an intuitive interface for users to input data (image, text, or numerical) and receive instant predictions.

---

## 🚀 Features

### 🔍 **Classification Models**
- **Gender Classification** – using Convolutional Neural Network (CNN) with image input.
- **Spam Mail Detection** – using Logistic Regression with text input.
- **Movie Recommendation System** – based on Cosine Similarity (text-based).
- **Titanic Survival Prediction** – using Logistic Regression.
- **Credit Card Approval Prediction** – using Logistic Regression.
- **Loan Approval Prediction** – using Support Vector Machine (SVM).
- **Rock vs Mine Prediction** – using Logistic Regression.
- **Diabetes Prediction** – using SVM.
- **Heart Disease Prediction** – using Logistic Regression.
- **Parkinson’s Disease Prediction** – using SVM.
- **Breast Cancer Prediction** – using Logistic Regression.
- **Wine Quality Prediction** – using Random Forest Regressor (treated as classification).

### 📈 **Regression Models**
- **Big Mart Sales Prediction** – using XGBoost Regressor.
- **Calories Burnt Prediction** – using XGBoost Regressor.
- **Car Price Prediction** – using Lasso Regression.
- **Gold Price Prediction** – using Random Forest Regressor.
- **House Price Prediction** – using XGBoost Regressor.
- **Medical Insurance Cost Prediction** – using Linear Regression.

---

## 🧩 Input Types

- 📷 **Image Input**:  
  - Gender classification (accepts image file upload)

- 📝 **Text Input**:  
  - Spam detection  
  - Movie recommendations

- 🔢 **Numerical Input**:  
  - All other models require numerical feature inputs via form fields

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Backend Models**:  
  - Scikit-learn (Logistic Regression, Random Forest, SVM, Lasso, Linear Regression)  
  - XGBoost  
  - TensorFlow / Keras (CNN for image classification)
- **Text Processing**: CountVectorizer, TfidfVectorizer, Cosine Similarity (for recommendation)

---

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

streamlit run all_in_one_frontend_ml.py


