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
```
├── data/
│   └── bodyPerformance.csv
├── models/
│   ├── model.joblib
│   └── scaler.joblib
├── notebooks/
│   └── body_performance_ml_project.ipynb
├── src/
│   └── train.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```
- **notebooks/**: Complete Jupyter Notebook containing Data Analysis (EDA), Data Preprocessing, and Model Comparisons.
- **src/train.py**: Standalone Python script to train the chosen MLP model and export the model & scaler.
- **app.py**: Interactive Streamlit web application for real-time predictions.

## 🚀 How to Run Locally

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/saeedsad821-ux/Body-Performance-ML-Project.git
   cd Body-Performance-ML-Project
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**:
   The pre-trained model is already included (`model.joblib`), but you can retrain it:
   ```bash
   python src/train.py
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   The app will be accessible at `http://localhost:8501`.

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Easiest)
Since the app uses Streamlit and is hosted on GitHub, you can deploy it for free with one click:
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub.
3. Click **New app** and select this repository.
4. Set the Main file path to `app.py` and click **Deploy**.

### Option 2: Docker Deployment
A `Dockerfile` is included for containerized deployment on any platform (AWS, GCP, Render, Railway, etc.).

**To run using Docker locally:**
```bash
docker build -t body-performance-app .
docker run -p 8501:8501 body-performance-app
```
The app will be live at `http://localhost:8501`.

## 🧠 Model Details
The final model chosen is a **Neural Network (MLPClassifier)** due to its consistency and high F1-score across various test splits. The features are standardized using `StandardScaler` prior to inference.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the MIT License.
