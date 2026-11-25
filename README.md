# ⚙️ Machine Failure Prediction Using Machine Learning

## 📌 Project Overview

Machine Failure Prediction is a Machine Learning project designed to predict whether an industrial machine is likely to fail based on its operational and environmental conditions.

The project analyzes machine parameters such as air temperature, process temperature, rotational speed, torque, tool wear, and machine type to identify potential machine failures.

The main goal is to develop a reliable classification model that can help identify machine failure risks and support predictive maintenance.

## 🎯 Project Objectives

- Predict whether a machine will fail or not.
- Analyze the factors responsible for machine failures.
- Perform Exploratory Data Analysis (EDA).
- Compare different Machine Learning algorithms.
- Improve model performance using hyperparameter tuning.
- Evaluate models using classification metrics.
- Build a machine failure prediction system.

## 📂 Dataset Information

The project uses an industrial machine failure prediction dataset.

### Features Used

| Feature | Description |
|---|---|
| Type | Machine type (L, M, H) |
| Air_temperature | Air temperature in Kelvin |
| Process_temperature | Process temperature in Kelvin |
| Rotational_speed | Machine rotational speed in RPM |
| Torque | Machine torque in Nm |
| Tool_wear | Tool wear time in minutes |

### Target Variable

- `Machine_failure`
  - 0 = No Failure
  - 1 = Failure

The notebook also removes the following failure-related columns to avoid target leakage:

- TWF
- HDF
- PWF
- OSF
- RNF

UDI and Product_ID are also excluded from model training.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab
- Jupyter Notebook

## 🔍 Exploratory Data Analysis (EDA)

The project performs several data analysis and visualization tasks:

1. Dataset information and statistical analysis.
2. Missing value detection.
3. Duplicate record detection.
4. Machine failure distribution analysis.
5. Box plots of machine parameters.
6. Histograms of numerical features.
7. Failure rate analysis by machine type.
8. Correlation heatmap.

## 🤖 Machine Learning Algorithms

Three Machine Learning classification algorithms are implemented:

### 1. Logistic Regression
Used as a baseline classification model to predict machine failure.

### 2. Decision Tree Classifier
Used to identify patterns and relationships between machine operating conditions and failures.

### 3. Random Forest Classifier
Uses multiple decision trees to improve classification performance and prediction reliability.

## ⚙️ Model Development Workflow

1. Import required libraries.
2. Load the dataset.
3. Clean and preprocess column names.
4. Perform Exploratory Data Analysis.
5. Separate features and target variable.
6. Apply StandardScaler to numerical features.
7. Apply OneHotEncoder to categorical features.
8. Split the dataset into training and testing sets (80:20).
9. Train three classification models.
10. Evaluate model performance.
11. Perform 5-fold Stratified Cross-Validation.
12. Apply GridSearchCV for hyperparameter tuning.
13. Select the best-performing model.
14. Save the trained model using Joblib.
15. Predict machine failure for new input data.

## 📊 Model Evaluation Metrics

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

The project uses F1-score as the main scoring metric for cross-validation and hyperparameter tuning.

## 🔧 Hyperparameter Tuning

GridSearchCV is used to optimize the performance of all three models.

The following parameters are tuned:

- Logistic Regression: C and class_weight.
- Decision Tree: max_depth, min_samples_split, criterion, and class_weight.
- Random Forest: n_estimators, max_depth, min_samples_split, and class_weight.

The best model is selected based on the highest cross-validation F1-score.

## 💾 Model Saving and Loading

The selected model is saved using Joblib.

```python
import joblib

joblib.dump(best_model, "machine_failure_model.joblib")
```

Load the saved model:

```python
model = joblib.load("machine_failure_model.joblib")
```

## 🔮 Machine Failure Prediction

The project includes a prediction function that accepts six machine parameters.

```python
def predict_failure(machine_type, air_temp, process_temp,
                    rpm, torque, tool_wear):

    new = pd.DataFrame([{
        "Type": machine_type,
        "Air_temperature": air_temp,
        "Process_temperature": process_temp,
        "Rotational_speed": rpm,
        "Torque": torque,
        "Tool_wear": tool_wear
    }])

    pred = model.predict(new)[0]
    prob = model.predict_proba(new)[0][1]

    return pred, prob
```

Example:

```python
pred, prob = predict_failure(
    "M", 300, 310, 1500, 40, 100
)

print("Machine Failure:",
      "Yes" if pred == 1 else "No")

print("Failure Probability:",
      f"{prob * 100:.2f}%")
```

## 📁 Project Structure

```text
Machine-Failure-Prediction/
│
├── Machine_Failure_Prediction.ipynb
│
├── dataset/
│   └── machine_failure_dataset.csv
│
├── machine_failure_model.joblib
│
├── requirements.txt
│
└── README.md
```

## 🚀 Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/Machine-Failure-Prediction.git
```

### Step 2: Navigate to the project folder

```bash
cd Machine-Failure-Prediction
```

### Step 3: Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### Step 4: Run Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Open the notebook

Open `Machine_Failure_Prediction.ipynb` and run the cells sequentially.

**Note:** The original notebook uses Google Colab's file upload functionality. If running locally, replace that section with the appropriate CSV file path.

## 🌟 Future Enhancements

- Develop a web application using Streamlit or Flask.
- Integrate real-time machine sensor data.
- Implement an automated predictive maintenance system.
- Improve model performance through advanced algorithms.
- Build an interactive dashboard for machine monitoring.
- Generate early warnings for machines at risk of failure.

## 👨‍💻 Author

SAMIRAN CHANAK

GitHub:https://github.com/samiran-user

## ⭐ Support

If you find this project useful, please give this repository a star ⭐ on GitHub!

---

*This project is developed for educational purposes and demonstrates the application of Machine Learning in industrial predictive maintenance.*
