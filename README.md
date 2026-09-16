# 🏠 Predictive Insight Engine

## 📌 Project Overview

**Predictive Insight Engine** is a machine learning project focused on **supervised learning and regression techniques** for predicting house prices.

The project uses a real-estate dataset containing **4,200 house records** and applies different regression approaches to understand how property characteristics influence house prices.

The project covers:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Gradient Descent
* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini-Batch Gradient Descent
* Model Performance Evaluation
* Bias-Variance Trade-Off
* Overfitting and Underfitting
* Model Diagnostics

---

## 🎯 Objective

The objective of this project is to understand, implement, and evaluate supervised learning algorithms, with a focus on regression techniques and model performance analysis.

### The project aims to explore:

* Linear Regression and its variants
* Gradient Descent optimization techniques
* Model performance evaluation
* Bias-Variance Trade-Off
* Overfitting and Underfitting
* Model behavior and diagnostic analysis

---

## 📊 Dataset

The project uses a real-estate house price dataset containing **4,200 records and 12 columns**.

### Dataset Features

| Feature                | Description                 |
| ---------------------- | --------------------------- |
| `house_id`             | Unique house identifier     |
| `area_sqft`            | House area in square feet   |
| `bedrooms`             | Number of bedrooms          |
| `bathrooms`            | Number of bathrooms         |
| `location_score`       | Location quality score      |
| `age_years`            | Age of the property         |
| `distance_city_km`     | Distance from city          |
| `lot_size_sqft`        | Size of the land/lot        |
| `has_garage`           | Garage availability         |
| `has_pool`             | Pool availability           |
| `renovation_years_ago` | Years since last renovation |
| `house_price_inr`      | House price in INR          |

### Target Variable

```text
house_price_inr
```

### Independent Variables Used

```text
area_sqft
bedrooms
bathrooms
location_score
age_years
```

---

## 🧠 Machine Learning Concepts Covered

### 1. Supervised Learning

Supervised learning algorithms are trained using labeled data. The model learns the relationship between input features `X` and a known target variable `y`.

### 2. Regression

Regression is used to predict continuous numerical values.

In this project, the target is:

```text
House Price
```

### 3. Simple Linear Regression

Simple Linear Regression uses one independent variable to predict the target variable.

In this project:

```text
Independent Variable:
area_sqft

Dependent Variable:
house_price_inr
```

The basic equation is:

```text
y = mx + c
```

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed using:

* `head()`
* `info()`
* `describe()`
* `shape`
* `isnull().sum()`
* Pairplot

### Dataset Shape

```text
Rows: 4200
Columns: 12
```

### Missing Values

The dataset contains **no missing values** in the analyzed columns.

---

## 📈 Feature Selection

The following features were selected for regression:

```python
features = [
    'area_sqft',
    'bedrooms',
    'bathrooms',
    'location_score',
    'age_years'
]

target = 'house_price_inr'
```

The input and target variables are created using:

```python
X = DF[features]
y = DF[target]
```

---

## ✂️ Train-Test Split

The dataset was divided into training and testing datasets using an **80:20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This gives the model separate data for training and evaluation.

---

# 📌 Part 1: Simple Linear Regression

Simple Linear Regression was implemented using only:

```text
area_sqft
```

### Model

```python
slr = LinearRegression()

slr.fit(X_train_slr, y_train)

y_pred_slr = slr.predict(X_test_slr)
```

### Analysis

A regression line was plotted between house area and house price.

Residual analysis was also performed to check model behavior and homoscedasticity.

---

# 📌 Part 2: Model Evaluation

The following evaluation metrics were used:

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

Lower MSE indicates lower prediction error.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and represents the error magnitude in the original target units.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

### R² Score

R² represents the proportion of variance in the target variable explained by the model.

### Adjusted R²

Adjusted R² accounts for the number of predictors used in the model.

---

## 🧮 Evaluation Function

```python
def evaluate_model(y_true, y_pred, p):
    n = len(y_true)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    adj_r2 = 1 - (1-r2) * (n-1) / (n-p-1)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adj_R2': adj_r2
    }
```

---

# 📌 Part 3: Multiple Linear Regression

Multiple Linear Regression uses multiple independent variables simultaneously.

The project uses:

```text
area_sqft
bedrooms
bathrooms
location_score
age_years
```

### Model

```python
mlr = LinearRegression()

mlr.fit(X_train, y_train)

y_pred_mlr = mlr.predict(X_test)
```

### Purpose

Multiple Linear Regression captures the effect of multiple property characteristics instead of relying only on house area.

According to the notebook analysis, model performance improved substantially compared with Simple Linear Regression.

---

# 📌 Part 4: Polynomial Regression

Polynomial Regression was used to capture possible non-linear relationships between the features and house price.

A degree-2 polynomial transformation was applied.

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()

poly_reg.fit(X_train_poly, y_train)

y_pred_poly = poly_reg.predict(X_test_poly)
```

### Degree 2 Polynomial Regression

The notebook reports approximately:

```text
R² Score: 0.96
```

The model showed improved performance compared with Multiple Linear Regression in the notebook's comparison.

---

# 📊 Model Comparison

The notebook compares Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression.

| Model         |          MAE |                   MSE |         RMSE |   R² | Adjusted R² |
| ------------- | -----------: | --------------------: | -----------: | ---: | ----------: |
| Simple LR     | 6,294,593.70 | 66,989,260,021,849.47 | 818,469,670* | 0.56 |        0.56 |
| Multiple LR   | 2,641,320.38 | 12,848,701,729,036.91 | 3,584,508.58 | 0.92 |        0.92 |
| Polynomial LR | 1,715,218.81 |  5,429,392,016,489.12 | 2,330,105.58 | 0.96 |        0.96 |

> **Note:** The SLR RMSE value above is reproduced from the notebook's comparison table. The notebook's evaluation function calculates RMSE as `sqrt(MSE)`, so this displayed SLR RMSE should be rechecked when presenting final results.

---

# 📌 Part 5: Gradient Descent

Gradient Descent is an optimization algorithm used to minimize a cost function.

The model parameters are updated iteratively in the opposite direction of the gradient.

The project implements three Gradient Descent variants:

1. Batch Gradient Descent
2. Stochastic Gradient Descent
3. Mini-Batch Gradient Descent

---

## ⚙️ Feature Scaling

StandardScaler was used before implementing Gradient Descent.

```python
scaler_x = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train)

y_train_scaled = (
    y_train - y_train.mean()
) / y_train.std()

X_b = np.c_[
    np.ones((len(X_train_scaled), 1)),
    X_train_scaled
]
```

---

# 📌 Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the entire training dataset before updating the parameters.

```python
def batch_gradient_descent(X, y, lr=0.01, epochs=50):

    m = len(y)

    theta = np.random.randn(X.shape[1])

    cost_history = []

    for epoch in range(epochs):

        gradients = (
            2/m *
            X.T.dot(
                X.dot(theta) - y
            )
        )

        theta = theta - lr * gradients

        cost_history.append(
            compute_cost(X, y, theta)
        )

    return theta, cost_history
```

---

# 📌 Stochastic Gradient Descent

Stochastic Gradient Descent updates model parameters using randomly selected individual observations.

```python
def stochastic_gradient_descent(X, y, lr=0.01, epochs=50):

    m = len(y)

    theta = np.random.randn(X.shape[1])

    cost_history = []

    for epoch in range(epochs):

        for i in range(m):

            random_index = np.random.randint(m)

            xi = X[
                random_index:random_index+1
            ]

            yi = (
                y.iloc[random_index:random_index+1]
                if isinstance(y, pd.Series)
                else y[random_index:random_index+1]
            )

            gradients = (
                2 *
                xi.T.dot(
                    xi.dot(theta) - yi
                )
            )

            theta = theta - lr * gradients

        cost_history.append(
            compute_cost(X, y, theta)
        )

    return theta, cost_history
```

---

# 📌 Mini-Batch Gradient Descent

Mini-Batch Gradient Descent uses a small batch of observations for each parameter update.

```python
def minibatch_gradient_descent(
    X,
    y,
    lr=0.01,
    epochs=50,
    batch_size=32
):

    m = len(y)

    theta = np.random.randn(X.shape[1])

    cost_history = []

    for epoch in range(epochs):

        shuffled_indices = np.random.permutation(m)

        X_shuffled = X[shuffled_indices]

        y_shuffled = (
            y.iloc[shuffled_indices].values
            if isinstance(y, pd.Series)
            else y[shuffled_indices]
        )

        for i in range(0, m, batch_size):

            xi = X_shuffled[
                i:i+batch_size
            ]

            yi = y_shuffled[
                i:i+batch_size
            ]

            gradients = (
                2/len(xi) *
                xi.T.dot(
                    xi.dot(theta) - yi
                )
            )

            theta = theta - lr * gradients

        cost_history.append(
            compute_cost(X, y, theta)
        )

    return theta, cost_history
```

---

# 📉 Gradient Descent Comparison

The notebook compares:

```text
Batch GD
Stochastic GD
Mini-Batch GD
```

### Batch Gradient Descent

* Smooth convergence
* Uses the entire dataset per update
* Computationally slower per epoch

### Stochastic Gradient Descent

* Faster individual parameter updates
* More noisy convergence
* Does not always settle smoothly at the exact minimum

### Mini-Batch Gradient Descent

* Combines advantages of Batch GD and SGD
* Faster than Batch GD
* More stable than SGD
* Provides a balance between speed and convergence stability

---

# 📌 Part 6: Bias-Variance Trade-Off

The project studies the relationship between model complexity, bias, and variance.

Three models are compared:

```text
Simple Linear Regression
Multiple Linear Regression
Polynomial Regression Degree 4
```

### Simple Linear Regression

The notebook identifies SLR as a **high-bias model** because it uses only house area and may not capture the complete relationship between property features and price.

### Multiple Linear Regression

MLR is described as having a more balanced bias-variance behavior because it uses multiple relevant features.

### Polynomial Regression Degree 4

The notebook uses Degree 4 Polynomial Regression to demonstrate **high variance / overfitting behavior**.

As model complexity increases:

```text
Training Error ↓
Bias ↓

But eventually:

Variance ↑
Test Error ↑
```

---

# 📌 Overfitting and Underfitting

## Underfitting

Underfitting occurs when a model is too simple to capture the underlying patterns in the data.

Example:

```text
Using only house area to predict house price
```

This can result in high bias.

## Overfitting

Overfitting occurs when a model becomes too complex and learns noise or random fluctuations from the training data.

Example:

```text
High-degree polynomial regression
```

The notebook notes that higher-degree polynomial models such as Degree 3 or above may show signs of overfitting.

---

# 📊 Technologies Used

### Programming Language

```text
Python
```

### Libraries

```text
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
```

### Main Scikit-learn Components

```python
train_test_split
LinearRegression
mean_squared_error
mean_absolute_error
r2_score
PolynomialFeatures
StandardScaler
```

---

# 📁 Project Structure

```text
Predictive-Insight-Engine/
│
├── Predictive Insight Engine .ipynb
│
├── RealEstate_HousePrice_Dataset_4200.csv
│
└── README.md
```

---

# 🚀 How to Run the Project

## 1. Clone or Download the Project

Download the project files to your local computer.

## 2. Install Required Libraries

Run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 3. Open the Notebook

Open:

```text
Predictive Insight Engine .ipynb
```

using Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

## 4. Add the Dataset

Make sure the dataset is located in the same directory as the notebook.

The notebook reads the dataset using:

```python
DF = pd.read_csv(
    "RealEstate_HousePrice_Dataset_4200 - RealEstate_HousePrice_Dataset_4200.csv.csv"
)
```

## 5. Run All Cells

Execute the notebook cells sequentially to reproduce:

* Dataset analysis
* Feature selection
* Train-test split
* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Model evaluation
* Gradient Descent
* Bias-Variance analysis
* Model comparisons

---

# 🔬 Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Descriptive Statistics
   ↓
Missing Value Check
   ↓
Feature Selection
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Simple Linear Regression
   ↓
Model Evaluation
   ↓
Multiple Linear Regression
   ↓
Polynomial Regression
   ↓
Gradient Descent
   ↓
Bias-Variance Analysis
   ↓
Model Diagnostics
   ↓
Final Analysis
```

---

# 💡 Key Findings

Based on the analysis documented in the notebook:

* The dataset contains **4,200 observations** and **12 columns**.
* No missing values were identified.
* Simple Linear Regression using only `area_sqft` explains approximately **56% of the target variance** according to the notebook.
* Multiple Linear Regression improves the reported R² to approximately **0.92**.
* Degree-2 Polynomial Regression reports an R² of approximately **0.96**.
* Multiple property characteristics provide more information about house prices than house area alone.
* Higher-degree polynomial models can introduce overfitting.
* Mini-Batch Gradient Descent provides a balance between convergence stability and computational efficiency.

---

# 🏢 Business Interpretation

A real-estate organization can use regression models to estimate house prices from property characteristics such as:

```text
Area
Bedrooms
Bathrooms
Location Score
Property Age
```

Such a model can support:

* Property price estimation
* Real-estate appraisal
* Market analysis
* Preliminary pricing decisions
* Data-driven property evaluation

The notebook specifically discusses using Multiple Linear Regression or a low-degree polynomial model for house-price estimation.

---

# 📚 Concepts Demonstrated

```text
Supervised Learning
        ↓
Regression
        ↓
Simple Linear Regression
        ↓
Multiple Linear Regression
        ↓
Polynomial Regression
        ↓
Gradient Descent
        ↓
Batch GD
        ↓
Stochastic GD
        ↓
Mini-Batch GD
        ↓
Model Evaluation
        ↓
Bias-Variance Trade-Off
        ↓
Overfitting
        ↓
Underfitting
        ↓
Model Diagnostics
```

---

# 👨‍💻 Author

**Man Trivedi**

Machine Learning / Data Science Project

---

# 📄 License

This project is intended for educational and learning purposes.
