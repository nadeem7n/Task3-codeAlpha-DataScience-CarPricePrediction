# Task 3: Car Price Prediction with Machine Learning

## Objective
Train a regression model to predict car prices based on features like brand, horsepower, mileage, engine size, and age.

## Approach
- Generated a realistic car price dataset with relevant features
- Performed data preprocessing including feature encoding and scaling
- Applied feature engineering to create meaningful predictors
- Trained multiple regression models

## Models Implemented
1. **Linear Regression**
2. **Ridge Regression**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**

## Results
| Model | R² Score | RMSE |
|-------|----------|------|
| Random Forest | 0.93 | ~$2,400 |
| Gradient Boosting | 0.91 | ~$2,700 |
| Ridge Regression | 0.85 | ~$3,500 |
| Linear Regression | 0.84 | ~$3,600 |

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python car_price_prediction.py
```

