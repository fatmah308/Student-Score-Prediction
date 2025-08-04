# 📚 Student Performance Prediction using Regression Models

This project demonstrates the use of **Linear Regression** and **Polynomial Regression** to analyze and predict students' exam scores based on the number of hours they studied. It uses a dataset named `StudentPerformanceFactors.csv` and visualizes the relationship between study time and performance.

---

## 🧠 Project Objective

To compare the performance of simple linear and polynomial regression models in predicting student exam scores from study hours, using Python and popular machine learning libraries.

---

## 📂 Dataset

The dataset used (`StudentPerformanceFactors.csv`) contains:
- `Hours_Studied`: Number of hours a student studied.
- `Exam_Score`: The score the student achieved on the exam.

---

## 📊 Features & Workflow

### 1. **Data Exploration**
- Display basic dataset structure using `.head()`, `.info()`, and `.describe()`.
- Visualize raw data with a scatter plot to understand the trend between study time and exam scores.
- Calculate Pearson correlation between `Hours_Studied` and `Exam_Score`.

### 2. **Linear Regression**
- Train a simple linear regression model.
- Evaluate using **R² Score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.
- Visualize predicted vs actual exam scores on a test set.

### 3. **Polynomial Regression (Bonus Task)**
- Apply polynomial transformation to capture non-linear relationships.
- Train and evaluate a polynomial regression model (degree=2).
- Visualize the polynomial curve fitted over actual data points.

### 4. **Model Comparison**
- Display a side-by-side comparison of both models in terms of R² Score and RMSE.

---

## 📈 Results

You will get:
- A better understanding of how well linear vs polynomial regression performs on this kind of data.
- Visual confirmation of model fit and generalizability.

---

## 🛠️ Requirements

Make sure the following libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
