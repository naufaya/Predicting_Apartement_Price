# 🏢 Daegu Apartment Price Prediction Project 📈

## Project Overview
This project aims to predict apartment prices in Daegu, South Korea, using a dataset that contains various features of the apartments such as size, proximity to subway stations, year built, and more. We use several machine learning models, including XGBoost, to identify the most significant factors that influence apartment prices and create accurate predictive models.

## 📂 Contents

1. Business Problem Understanding
2. Data Understanding
3. Data Preprocessing
4. Modeling
5. Conclusion
6. Recommendations

## 🌐 Data Source
The dataset is from Daegu, South Korea, detailing apartment listings with various internal and external attributes such as proximity to subway stations and apartment facilities. [Dataset Link](https://drive.google.com/drive/folders/13bd2xHVkKwbrZrItJUm0LcH7NGaonm-0)

---

## 💼 1. Business Problem Understanding

### Context
Daegu, South Korea, is experiencing rapid urbanization and population growth, leading to increasing demand for apartments. The prices of these apartments are influenced by both **internal** (year built, size, facilities) and **external** factors (proximity to transportation and public services). The challenge is to predict apartment prices accurately to help property owners and investors set the right selling price.

### Problem Statement
In a competitive market, it’s essential for property owners and investors to set an optimal selling price. Too high, and the property will remain unsold; too low, and profits will be lost. The goal is to build a predictive model to assist in setting an accurate price based on factors like apartment size, hallway type, and accessibility to nearby services.

### Analytical Approach
A regression model will be built to predict apartment prices based on various internal and external factors. The primary evaluation metrics for model performance will be RMSE, MAE, and MAPE.

---

## 🔍 2. Data Understanding

### Dataset Information
The dataset contains listings of apartments in Daegu, with attributes such as:
- **HallwayType**: Type of apartment hallway
- **TimeToSubway**: Time taken to reach the nearest subway station
- **SubwayStation**: Name of the nearest subway station
- **N_FacilitiesNearBy (ETC/Public Office/University)**: Number of nearby facilities
- **N_Parkinglot (Basement)**: Number of parking spaces available in the basement
- **YearBuilt**: The year the apartment was constructed
- **Size (sqf)**: Apartment size in square feet
- **SalePrice**: Apartment sale price in Korean Won (₩)

### Exploratory Data Analysis (EDA)
- **Correlation Analysis**: A correlation heatmap was created to explore relationships between numerical features and SalePrice. Key insights include:
  - **Size (sqf)** has the strongest positive correlation with SalePrice.
  - **YearBuilt** has a negative correlation, indicating that newer apartments are generally priced higher.
  
- **Bivariate Analysis**: Scatter plots between numerical features and SalePrice revealed strong relationships between apartment size and price, while features like proximity to subway stations had weaker relationships.

### Data Insights
- Larger apartments are more expensive, while older apartments are generally cheaper.
- Proximity to public transportation and the number of nearby facilities slightly affects apartment prices.

---

## 🔧 3. Data Preprocessing

### Steps Involved:
- **Handling Outliers**: Outliers were identified in the `Size(sqf)` and `SalePrice` columns and retained since they represent real market prices.
- **Encoding Categorical Variables**: 
  - One-hot encoding was applied to features like `HallwayType` and `SubwayStation`.
  - Ordinal encoding was applied to `TimeToSubway` to capture its ordered nature.
  
- **Feature Scaling**: Robust scaling was applied to numerical features to normalize the data without being heavily influenced by outliers.

### Data Splitting
The dataset was split into training (80%) and testing (20%) sets for model evaluation.

---

## 🚀 4. Modeling

### Models Used
Several machine learning models were tested to predict apartment prices:
- **Linear Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Lasso and Ridge Regression**
- **XGBoost**

### Model Evaluation Metrics
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

### Benchmark Model Results
After cross-validation:
- **XGBoost** emerged as the best-performing model with the lowest RMSE, MAE, and MAPE scores.
- **Random Forest** and **Decision Tree** also performed well but were slightly less accurate than XGBoost.

### Hyperparameter Tuning
The XGBoost model was further improved by tuning hyperparameters using **RandomizedSearchCV**.

### Final Model Performance
After tuning, the **XGBoost** model achieved a slight improvement compared to the untuned model, confirming the effectiveness of hyperparameter tuning.

---

## 🏆 5. Conclusion

Based on the model analysis:
- **Size** is the most significant factor in determining apartment prices in Daegu.
- Apartments with more **facilities** and **better parking availability** are valued higher.
- **HallwayType** (especially terraced-type hallways) significantly influences apartment prices.
- Closer proximity to subway stations and newer buildings also positively affect apartment prices.

---

## 📊 6. Recommendations

1. **Additional Features**: Consider adding more relevant features such as the number of rooms, distance to business districts, or additional public transport options.
2. **Updated Data**: Using more recent data would improve the model’s predictive power, especially after events like economic shifts or the pandemic.
3. **Complex Models**: For larger datasets, more complex models like neural networks could be explored to further improve accuracy.
4. **Sentiment Analysis**: If buyer reviews are available, incorporating sentiment analysis could add a valuable layer of insights for future models.
5. **A/B Testing**: Implement A/B testing to see how well the model performs in real-world pricing scenarios.

---

## 💾 7. Saving the Model

The final XGBoost model is saved using `pickle` for easy deployment and future use:
```python
import pickle

# Save model
pickle.dump(xgb_tuning, open('Model_Daegu_Apartment_XGB.sav', 'wb'))

# Load model
loaded_model = pickle.load(open('Model_Daegu_Apartment_XGB.sav', 'rb'))
```

---

## 👩‍💻 About the Author
This project was crafted with care by [Namira R.D]. 🌈
