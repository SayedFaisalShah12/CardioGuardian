# CardioGuardian - Heart Disease Prediction System

A production-level Machine Learning project for predicting heart disease using clinical parameters. This project implements a complete ML pipeline with data preprocessing, model training, evaluation, and a user-friendly Streamlit web application.

Links:
Streamlit: https://sayedfaisalshah12-cardioguardian-app-xvau59.streamlit.app/
HuggingFace: https://huggingface.co/spaces/Sayed-Shah/CardioGuardian

## 🎯 Project Overview

CardioGuardian is a comprehensive ML system that:
- Preprocesses heart disease data with proper handling of missing values and feature scaling
- Compares multiple machine learning models to find the best performer
- Provides comprehensive evaluation metrics
- Saves the best model for production use
- Offers an interactive web interface for predictions

## 📁 Project Structure

```
CardioGuardian/
│
├── heart.csv                    # Dataset file
├── data_preprocessing.py        # Data loading, cleaning, and preprocessing
├── train_model.py              # Model training and comparison
├── evaluate_model.py           # Model evaluation with metrics
├── predict.py                  # Prediction functions
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Saved models (created after training)
│   ├── best_model.pkl
│   └── scaler.pkl
│
└── results/                    # Evaluation results (created after evaluation)
    ├── confusion_matrix.png
    └── roc_curve.png
```

## 🚀 Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Train and save the best model:

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Compare multiple ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree)
- Select the best model based on accuracy
- Save the best model and scaler to the `models/` directory

### 3. Evaluate the Model

Evaluate the trained model with comprehensive metrics:

```bash
python evaluate_model.py
```

This will display:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score (if available)
- Generate visualization plots

### 4. Run the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser where you can:
- Input patient clinical parameters
- Get real-time heart disease predictions
- View prediction probabilities

## 📊 Dataset

The `heart.csv` dataset contains the following features:

- **age**: Age in years
- **sex**: Sex (0 = Female, 1 = Male)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (0/1)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (0/1)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-4)
- **thal**: Thalassemia (0-3)
- **target**: Target variable (0 = No disease, 1 = Disease)

## 🔧 Module Details

### data_preprocessing.py
- Loads data from CSV
- Handles missing values (median for numerical, mode for categorical)
- Splits data into train/test sets (80/20)
- Applies StandardScaler for feature scaling
- Saves/loads scaler for consistent preprocessing

### train_model.py
- Compares 6 different ML algorithms
- Selects best model based on accuracy
- Saves the best model using pickle
- Provides model loading functionality

### evaluate_model.py
- Calculates comprehensive metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
  - Confusion Matrix
- Generates visualization plots
- Provides detailed classification reports

### predict.py
- Makes predictions for single samples or batches
- Handles feature scaling automatically
- Returns predictions with probabilities
- Easy-to-use API for integration

### app.py
- Interactive Streamlit web interface
- User-friendly input forms
- Real-time predictions
- Visual probability displays
- Professional UI design

## 📈 Model Performance

The system compares the following models:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine
5. K-Nearest Neighbors
6. Decision Tree Classifier

The best model is automatically selected and saved based on test accuracy.

## 🎨 Features

- ✅ Production-ready code structure
- ✅ Comprehensive data preprocessing
- ✅ Multiple model comparison
- ✅ Detailed evaluation metrics
- ✅ Model persistence (pickle)
- ✅ Interactive web interface
- ✅ Well-documented code
- ✅ Error handling

## ⚠️ Important Disclaimer

**This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.**

## 📝 Usage Examples

### Making a Prediction Programmatically

```python
from predict import predict_from_dict

features = {
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 0,
    'ca': 0,
    'thal': 1
}

result = predict_from_dict(features)
print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability']}")
```

### Loading and Using the Model

```python
from train_model import load_model
from data_preprocessing import load_scaler
from predict import predict_single

model = load_model('models/best_model.pkl')
scaler = load_scaler('models/scaler.pkl')

# Make prediction
result = predict_single(model, scaler, features)
```

## 🔍 Evaluation Metrics

The evaluation module provides:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (if probabilities available)
- **Confusion Matrix**: Detailed breakdown of predictions

## 🤝 Contributing

This is a production-level ML project template. Feel free to:
- Add more models
- Improve preprocessing
- Enhance the web interface
- Add more evaluation metrics

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

CardioGuardian ML Project

---

**Built with ❤️ for Heart Disease Prediction**

