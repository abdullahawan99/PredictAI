from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from builtins import sum
import requests
from datetime import datetime
from sklearn.metrics import r2_score


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/temperature", methods=["GET", "POST"])
def temperature():
    if request.method == "POST":
        import time
        time.sleep(1)

        print("[DEBUG] POST request received")

        if "datafile" in request.files:

            file = request.files.get("datafile")

            if file and file.filename.endswith(".csv"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)

                try:
                    df = pd.read_csv(filepath)

                    if "Temperature" not in df.columns:

                        return render_template("upload.html", error="CSV must include a 'Temperature' column.")

                    table = df.head().to_html(classes="table table-bordered text-center", index=False)

                    avg_temp = df["Temperature"].mean()
                    max_temp = df["Temperature"].max()
                    min_temp = df["Temperature"].min()

                    temps = df["Temperature"].values

                    if len(temps) >= 6:
                        X, y = [], []
                        for i in range(len(temps) - 5):
                            X.append(temps[i:i+5])
                            y.append(temps[i+5])

                        model = LinearRegression()
                        model.fit(X, y)

                        recent_days = temps[-5:]
                        tomorrow_temp = round(model.predict([recent_days])[0], 2)

                    else:
                        tomorrow_temp = "Not enough data for prediction"

                    
                    dates = df["Date"].astype(str).tolist() if "Date" in df.columns else None
                    temps_list = df["Temperature"].tolist()

                    return render_template("preview.html", table=table,
                                           avg_temp=avg_temp,
                                           max_temp=max_temp,
                                           min_temp=min_temp,
                                           total_rain=None,
                                           tomorrow_temp=tomorrow_temp,
                                           dates=dates,
                                           temps=temps_list,
                                           rain=None)

                except Exception as e:

                    return render_template("upload.html", error=f"Failed to process file: {e}")

            else:

                return render_template("upload.html", error="Please upload a valid .csv file.")

        elif all(f"temp{i}" in request.form for i in range(1, 6)):
            try:
                temps = []
                for i in range(1, 6):
                    val = request.form.get(f"temp{i}")
                    if not val or not val.replace('.', '', 1).isdigit():
                        return render_template("upload.html", error="All manual inputs must be numeric.")
                    temps.append(float(val))
                temps = temps[::-1]

                today = pd.Timestamp.today()
                dates = [(today - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                         for i in range(4, -1, -1)]

                df = pd.DataFrame({
                    "Date": dates,
                    "Temperature": temps
                })

                table = df.to_html(classes="table table-bordered text-center", index=False)

                avg_temp = df["Temperature"].mean()
                max_temp = df["Temperature"].max()
                min_temp = df["Temperature"].min()

                X_train = [[10, 12, 14, 16, 18], [11, 13, 15, 17, 19], [12, 14, 16, 18, 20]]
                y_train = [20, 21, 22]

                model = LinearRegression()
                model.fit(X_train, y_train)

                prediction = round(model.predict([temps])[0], 2)

                return render_template("preview.html",
                                       table=table,
                                       avg_temp=avg_temp,
                                       max_temp=max_temp,
                                       min_temp=min_temp,
                                       total_rain=None,
                                       tomorrow_temp=prediction,
                                       dates=dates,
                                       temps=temps,
                                       rain=None)

            except Exception as e:
                return render_template("upload.html", error=f"Error in manual prediction: {e}")

    return render_template("upload.html")


def get_category_from_product(product_name):
    product_name = str(product_name).lower()

    category_keywords = {
        "electronics": ["laptop", "phone", "tablet", "tv", "camera"],
        "clothing": ["shirt", "jeans", "jacket", "t-shirt", "dress"],
        "cosmetics": ["cream", "shampoo", "lotion", "perfume"],
        "books": ["book", "novel", "manual"],
        "furniture": ["sofa", "table", "chair", "desk"],
        "grocery": ["milk", "bread", "egg", "rice", "flour"]
    }

    for category, keywords in category_keywords.items():
        if any(keyword in product_name for keyword in keywords):
            return category.capitalize()

    return "Other"


@app.route("/sales", methods=["GET", "POST"])
def sales():
    if request.method == "POST":
        file = request.files.get("datafile")
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)

                currency_symbol = ''

                if 'Currency' not in df.columns:
                    raise ValueError(
                        "CSV must contain a 'Currency' column with the symbol (e.g., $, USD, EUR) in one row.")

                non_empty_currency = df['Currency'].dropna().astype(str)

                if non_empty_currency.empty:
                    raise ValueError(
                        "'Currency' column is present but empty. Please provide a currency symbol or name in one row.")

                unique_currencies = non_empty_currency.str.strip().str.upper().unique()

                if len(unique_currencies) > 1:
                    raise ValueError(
                        "Multiple currencies detected. Please use only one currency across all rows.")

                raw_currency = unique_currencies[0]
                currency_map = {
                    'USD': '$',
                    'EUR': '€',
                    'GBP': '£',
                    'INR': '₹',
                    'JPY': '¥'
                }
                currency_symbol = currency_map.get(raw_currency, raw_currency)

                required_columns = {"Date", "Product", "Price", "Sales", "Promo"}
                if not required_columns.issubset(df.columns):
                    raise ValueError("CSV must contain: Date, Product, Price, Sales, Promo")

                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")

                if df["Date"].duplicated().any():
                    raise ValueError(
                        "Duplicate dates found. Please ensure each date appears only once in the data.")

                full_date_range = pd.date_range(
                    start=df["Date"].min(), end=df["Date"].max(), freq="D")
                missing_dates = full_date_range.difference(df["Date"])

                if not missing_dates.empty:
                    raise ValueError(
                        f"Missing dates detected: {missing_dates.strftime('%Y-%m-%d').tolist()}. Please upload continuous daily data.")

                if df["Price"].nunique() > 1:
                    raise ValueError(
                        "Inconsistent pricing detected. Please ensure all rows use the same price.")

                if df["Product"].nunique() > 1:
                    raise ValueError(
                        "CSV contains multiple product names. Please upload data for only one product at a time.")

                if len(df) < 15:
                    raise ValueError(
                        "Please upload a CSV with at least 15 rows for reliable prediction.")

                df["DayOfWeek"] = df["Date"].dt.weekday

                max_lag = min(5, len(df) // 2)
                lags_to_use = list(range(1, max_lag + 1))

                for lag in lags_to_use:
                    df[f"lag_{lag}"] = df["Sales"].shift(lag)

                df = df.dropna().reset_index(drop=True)

                feature_cols = [f"lag_{lag}" for lag in lags_to_use] + \
                    ["Price", "Promo", "DayOfWeek"]

                X_train = df[feature_cols]
                y_train = df["Sales"]

                Q1 = y_train.quantile(0.25)
                Q3 = y_train.quantile(0.75)
                IQR = Q3 - Q1
                mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)

                X_train = X_train[mask]
                y_train = y_train[mask]

                if X_train.shape[0] == 0:
                    raise ValueError(
                        "Not enough data after preprocessing to train the model. Please upload more data.")

                from statsmodels.tsa.arima.model import ARIMA
                import xgboost as xgb
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import r2_score

                if df.shape[0] < 20:
                    raise ValueError(
                        "Not enough usable data after preprocessing to train the hybrid model. Please upload a CSV with at least 30 original rows containing real data.")

                try:
                    arima_model = ARIMA(df["Sales"], order=(5, 1, 0))
                    arima_result = arima_model.fit()
                    arima_forecast = float(arima_result.forecast(steps=1).iloc[0])
                except Exception as e:
                    raise ValueError(
                        "ARIMA model failed to fit. Please ensure data is consistent and not too short.")

                arima_fitted = arima_result.fittedvalues.shift(1).reset_index(drop=True)

                X_train = X_train.reset_index(drop=True)
                arima_fitted = arima_fitted.iloc[-len(X_train):].reset_index(drop=True)
                X_train['arima_forecast'] = arima_fitted
                feature_cols_with_arima = feature_cols + ['arima_forecast']

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                             n_estimators=100, max_depth=3, random_state=42)
                xgb_model.fit(X_train_scaled, y_train.iloc[-len(X_train):])

                last_row = df.iloc[-1]
                next_day_of_week = (last_row["DayOfWeek"] + 1) % 7
                sales_lags = [df[f"lag_{lag}"].iloc[-1] for lag in lags_to_use]
                price = last_row["Price"]
                promo = last_row["Promo"]
                X_input = sales_lags + [price, promo, next_day_of_week, arima_forecast]
                print("X_input (tomorrow):", X_input)
                print("Expected number of features:", len(feature_cols) + 1)

                input_df = pd.DataFrame([X_input], columns=feature_cols + ['arima_forecast'])

                X_input_scaled = scaler.transform(input_df)

                tomorrow_sales = round(xgb_model.predict(X_input_scaled)[0], 2)

                train_preds = xgb_model.predict(X_train_scaled)
                accuracy = round(r2_score(y_train.iloc[-len(X_train):], train_preds), 3)
                accuracy_percent = round(accuracy * 100, 2)

                importances = pd.Series(xgb_model.feature_importances_,
                                        index=feature_cols_with_arima)

                importances = importances.sort_values(ascending=False)

                preview_df = df[["Date", "Sales"]].tail(15)
                preview_df["Date"] = preview_df["Date"].dt.strftime("%Y-%m-%d")
                dates = preview_df["Date"].tolist()
                sales = preview_df["Sales"].tolist()

                df['Date'] = pd.to_datetime(df['Date'])
                last_date = df['Date'].max()
                start_date = last_date - pd.Timedelta(days=14)
                last_15_days_df = df[(df['Date'] >= start_date) & (df['Date'] <= last_date)]
                units_sold_last_15_days = last_15_days_df['Sales'].sum()

                max_sales_row_15_days = last_15_days_df.loc[last_15_days_df['Sales'].idxmax()]
                max_sales_date_15_days = max_sales_row_15_days['Date'].strftime("%Y-%m-%d")
                max_sales_value_15_days = max_sales_row_15_days['Sales']
                min_sales_row_15_days = last_15_days_df.loc[last_15_days_df['Sales'].idxmin()]
                min_sales_date_15_days = min_sales_row_15_days['Date'].strftime("%Y-%m-%d")
                min_sales_value_15_days = min_sales_row_15_days['Sales']

                product = df["Product"].iloc[0]
                category = get_category_from_product(product)
                recent_df = df[["Sales", "Price"]].tail(15)
                total_revenue = round((recent_df["Sales"] * recent_df["Price"]).sum(), 2)

                last_15_sales = preview_df["Sales"].tolist()
                avg_sales = round(sum(last_15_sales) / len(last_15_sales), 2)
                max_sales = max(last_15_sales)
                min_sales = min(last_15_sales)

                predicted_date = (pd.to_datetime(
                    preview_df["Date"].iloc[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                table = preview_df.to_html(classes="table table-bordered", index=False)

                today_sales = df['Sales'].iloc[-1]
                percentage_change = round(((tomorrow_sales - today_sales) / today_sales) * 100, 2)
                trend_icon = "fa-arrow-up" if percentage_change >= 0 else "fa-arrow-down"
                trend_label = "increase" if percentage_change >= 0 else "decrease"

                return render_template(
                    "sales_preview.html",
                    table=table,
                    avg_sales=avg_sales,
                    max_sales=max_sales,
                    min_sales=min_sales,
                    tomorrow_sales=tomorrow_sales,
                    dates=dates,
                    sales=sales,
                    product=product,
                    category=category,
                    price=price,
                    total_revenue=total_revenue,
                    currency_symbol=currency_symbol,
                    units_sold_last_15_days=units_sold_last_15_days,
                    max_sales_date_15_days=max_sales_date_15_days,
                    max_sales_value_15_days=max_sales_value_15_days,
                    min_sales_date_15_days=min_sales_date_15_days,
                    min_sales_value_15_days=min_sales_value_15_days,
                    predicted_sales=tomorrow_sales,
                    accuracy_percent=accuracy_percent,
                    today_sales=today_sales,
                    percentage_change=abs(percentage_change),
                    trend_icon=trend_icon,
                    trend_label=trend_label,
                    predicted_date=predicted_date,
                    importances=importances.to_dict(),
                    accuracy=accuracy,
                    model_used="Hybrid (ARIMA + XGBoost)",
                    arima_prediction=round(arima_forecast, 2)
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return render_template("upload_sales.html", error="Error: " + str(e))

    return render_template("upload_sales.html")


@app.route('/auto_predict')
def auto_predict():
    country = request.args.get('country')
    city = request.args.get('city')
    if not country or not city:
        return render_template('upload.html', error="Please enter both country and city.")

    try:

        VC_API_KEY = "RTT2S4BWSZVPYFJXLTADWBCY9"

        end_date = datetime.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=4)

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        vc_url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{city},{country}/{start_date_str}/{end_date_str}"
            f"?unitGroup=metric&key={VC_API_KEY}&include=days&elements=datetime,temp"
        )

        vc_response = requests.get(vc_url)
        vc_data = vc_response.json()

        temps = [day['temp'] for day in vc_data['days']]

        past_labels = [datetime.strptime(
            day['datetime'], "%Y-%m-%d").strftime('%a %d') for day in vc_data['days']]

        past_temps = [round(t, 2) for t in temps]

        X = np.array(range(5)).reshape(-1, 1)
        y = np.array(temps)
        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)
        confidence = round(r2 * 100, 2)

        prediction = model.predict(np.array([[5]]))[0]

        date_today = datetime.now()
        next_5_days_labels = [(date_today + timedelta(days=i)).strftime('%a %d')
                              for i in range(1, 6)]
        next_5_days_temps = [round(model.predict(np.array([[i]]))[0], 2) for i in range(5, 10)]

        API_KEY = "7e08bfeb509d8acdfe18a948f966783b"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            raise ValueError(data.get("message", "Unknown error"))

        temp = data["main"]["temp"]
        temp_min = data["main"]["temp_min"]
        temp_max = data["main"]["temp_max"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        wind_deg = data["wind"].get("deg", "N/A")
        wind_gust = data["wind"].get("gust", "N/A")
        pressure = data["main"]["pressure"]
        feels_like = data["main"]["feels_like"]

        if temp_max == temp_min:
            temp_max += 1

        if temp_max != temp_min:
            range_width = ((temp - temp_min) / (temp_max - temp_min)) * 100
        else:
            range_width = 50

        date_today = datetime.today().strftime('%A, %B %d, %Y')

        formatted_date = datetime.today().strftime('%A, %B %d, %Y')

        return render_template("auto_preview.html",
                               city=city.title(),
                               country=country.title(),
                               temp=temp,
                               feels_like=feels_like,
                               temp_min=temp_min,
                               temp_max=temp_max,
                               humidity=humidity,
                               wind_speed=wind_speed,
                               wind_deg=wind_deg,
                               wind_gust=wind_gust,
                               pressure=pressure,
                               prediction=round(prediction, 2),
                               confidence=confidence,
                               date_today=formatted_date,
                               next_5_days_labels=next_5_days_labels,
                               next_5_days_temps=next_5_days_temps,
                               past_labels=past_labels,
                               past_temps=past_temps,
                               range_width=round(range_width, 2)
                               )

    except Exception as e:
        return render_template('upload.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
