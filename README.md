# 🧠 E-Commerce Product Classification using TF-IDF + Machine Learning

This project performs **multi-class text classification** on an **e-commerce dataset** to classify product categories using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Machine Learning**. The classification is done using models like **Multinomial Naive Bayes**, **Logistic Regression**, and **Support Vector Machine (SVM)**. Text is preprocessed and vectorized using TF-IDF for improved performance and accuracy.

---

## 🔧 Technologies Used

- Python  
- scikit-learn  
- pandas  
- spaCy  
- joblib  
- Jupyter Notebook  
- TfidfVectorizer (from scikit-learn)  
- MultinomialNB, LogisticRegression, SVC (from scikit-learn)

---

## 📁 Dataset

The dataset contains **product names** from four categories:

- **Books**  
- **Clothing_Accessories**  
- **Electronics**  
- **Household**  

📌 The raw data was preprocessed in a separate notebook: `ecommerceDataset.ipynb`

### ✨ Preprocessing Steps with spaCy:
- Lemmatization  
- Stopword removal  
- Punctuation removal  
- Lowercasing  

Resulting in a clean dataset: `processed_ecommerceDataset.csv`, ready for model training.

📥 **Dataset Link:** [Download from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

---

## 📌 Key Features

- **TF-IDF** for feature extraction and vectorization of text data.
- Text cleaned and normalized using `spaCy`.
- Models trained:
  - Multinomial Naive Bayes  
  - Logistic Regression  
  - Linear Support Vector Machine (SVM)
- Evaluation done using accuracy, classification report, and confusion matrix.
- Best-performing model saved using `joblib`.
- Allows predictions on custom product descriptions.

---

## ✅ Results & Observations

| Model                  | Accuracy  | Best F1-Score Category |
|------------------------|-----------|-------------------------|
| Multinomial Naive Bayes | 92%     | Household               |
| Logistic Regression     | 95%     | Clothing_Accessories    |
| **Linear SVM** (Best)   | **96%** | **Clothing_Accessories**|

📌 The **SVM** model performed the best and was saved as `best_model_svm.pkl`.

---

## 📁 Project Files & Folders Explained

| File/Folder                 | Description                                           |
|----------------------------|-------------------------------------------------------|
| `ecommerceDataset.ipynb`   | Notebook for preprocessing the raw dataset using spaCy |
| `tfidf_ecommerceDataset.ipynb` | Main notebook for training, evaluation, and testing |
| `processed_ecommerceDataset.csv` | Cleaned dataset after text preprocessing           |
| `best_model_svm.pkl`       | Final saved model using joblib                       |
| `README.md`                | Project overview and usage instructions              |

---

## 🙌 Credits

Project by **Yahan Madhuhansa**  
This project showcases how **TF-IDF** and popular **machine learning models** can be used to solve a real-world **text classification** problem in the e-commerce domain.

