# 🏋️‍♂️ Body Performance ML Project

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end Machine Learning project to predict body performance classes based on physical measurements.

## 📌 Project Overview
This project takes physical metrics (such as age, height, weight, body fat %, grip strength, sit-ups count, etc.) and uses a trained **Multilayer Perceptron (MLP) Neural Network** classifier to predict a person's performance class (A, B, C, or D).

- **Class A:** Best Performance
- **Class B:** Good Performance
- **Class C:** Average Performance
- **Class D:** Poor Performance

## 📂 Repository Structure
- `body_performance_ml_project.ipynb`: Complete Jupyter Notebook containing Data Analysis (EDA), Data Preprocessing, and Model Comparisons (KNN, SVM, Decision Trees, Neural Networks).
- `train.py`: Standalone Python script to train the chosen MLP model and export the model & scaler as `.joblib` files.
- `app.py`: Interactive Streamlit web application for real-time predictions.
- `requirements.txt`: Python package dependencies.
- `bodyPerformance.csv`: The dataset used for training.

## 🚀 How to Run Locally

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/body-performance-ml.git
   cd body-performance-ml
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**:
   The pre-trained model is already included (`model.joblib`), but you can retrain it:
   ```bash
   python train.py
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   The app will be accessible at `http://localhost:8501`.

## 🧠 Model Details
The final model chosen is a **Neural Network (MLPClassifier)** due to its consistency and high F1-score across various test splits. The features are standardized using `StandardScaler` prior to inference.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the MIT License.
