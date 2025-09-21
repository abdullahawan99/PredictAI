#  CS50 Final Project: PredictAI

##  Student
**Name:** Muhammad Abdullah Awan
**Course:** CS50: Introduction to Computer Science
**Project Title:** PredictAI

---

#### Video link:  https://youtu.be/9yiNKaohF9E

---
##  Description

This project is a full-stack predictive analytics dashboard built with **Flask** that helps users forecast:

- **Sales Prediction** using a **Hybrid Machine Learning model (ARIMA + XGBoost)**
- **Temperature Trends** using **Linear Regression** with three user-friendly input modes:
  - Real-time weather API
  - Manual entry
  - CSV upload

The project combines **data preprocessing**, **feature engineering**, and **predictive modeling** with an interactive and clean **web interface**. It showcases how machine learning can enhance decision-making in sales planning and weather forecasting.

---

##  Project Goals

- Apply core concepts of **CS50** (web, Flask, Python, HTML/CSS/JS)
- Use **real-world data and APIs** to build predictive models
- Explore hybrid ML models and time series forecasting
- Create a modern **dashboard interface** for non-technical users
-  Solve real-world problems like sales forecasting and temperature prediction using AI


---

##  Features

###  Sales Prediction (Hybrid ML: ARIMA + XGBoost)
- **Input**: CSV with columns: `Date`, `Product`, `Price`, `Sales`, `Promo`, `Currency`
- **Feature Engineering**:
  - Lag features
  - Day of the week
- **Modeling**:
  - `ARIMA` captures time series trend/seasonality
  - `XGBoost` models multivariate patterns (price, promo, etc.)
  - Final prediction = average of both model outputs
- **Output**:
  - Predicted Tomorrow sales
  - Sales statistics (Min, Max, Avg)
  - Model R² Score
  - Total Revenue
  - Interactive sales chart with Chart.js
  - Feature Importance Chart

---

###  Temperature Prediction (Linear Regression)

####  1. Auto Prediction (API)
- **Input**: User enters city and country
- **Data Source**:
  - Last 5 days via **Visual Crossing API**
  - Current day via **OpenWeatherMap API**
- **Prediction**: Linear Regression on historical temperatures
- **Output**:
  - Tomorrow Temperature Prediction
  - Current temperature details
  - Wind speed, Wind Direction, Gusts, Humidity, Pressure
  - Min/Max/Avg stats
  - Forecast chart

####  2. Manual Input
- **Input**: User manually enters temperatures for the last 5 days
- **Prediction**: Linear Regression
- **Output**:
  - Predicted tomorrow's temperature
  - Stats (Min, Max, Avg)
  - Forecast chart

####  3. CSV Upload
- **Input**: CSV with `Date`, `Temperature` columns
- **Prediction**: Linear Regression
- **Output**:
  - Predicted Temperature for the next day
  - Stats summary
  - Forecast chart

---

##  Tech Stack

- **Backend**: Python 3.13, Flask
- **Data**: Pandas, NumPy
- **Machine Learning**:
  - `ARIMA`: from `statsmodels`
  - `XGBoost`: advanced regression
  - `scikit-learn`: linear regression, preprocessing
- **Frontend**: HTML5, Bootstrap 5, Jinja2
- **Visualization**: Chart.js, Matplotlib
- **APIs**:
  - [Visual Crossing Weather API](https://www.visualcrossing.com/)
  - [OpenWeatherMap API](https://openweathermap.org/)

---



## Installation
```bash
pip install -r requirements.txt
```





## requirements.txt
```bash
Flask==2.3.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
statsmodels==0.14.4
xgboost==3.0.2
requests==2.32.3
matplotlib==3.8.4
```



##  Status
```bash
 Feature

 Sales Prediction (Hybrid ML)
 Temperature Prediction (All 3 Modes)
 CSV Uploads + Validation
 Dashboard UI/UX
 API Integration
 ML Model Deployment
```

---


## Acknowledgments

- Harvard University — CS50

- Visual Crossing Weather API

- OpenWeatherMap API

- XGBoost, scikit-learn, statsmodels
