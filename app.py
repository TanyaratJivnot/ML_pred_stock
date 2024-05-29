from flask import Flask, jsonify
import joblib
import numpy as np
from pymongo import MongoClient
import requests
from datetime import datetime
import pytz
import logging

app = Flask(__name__)

# ตั้งค่า logging
logging.basicConfig(filename='prediction.log', level=logging.INFO, format='%(asctime)s %(message)s')

# โหลดโมเดล
model_paths = {
    1: 'models/Cocoglass.pkl',
    2: 'models/Coco.pkl',
    3: 'models/Coco_Buttle.pkl',
    4: 'models/Coco_Pudding.pkl',
    6: 'models/Coco_Meat.pkl',
    10: 'models/Corn_milk.pkl',
    14: 'models/Big_popcorn.pkl',
    15: 'models/meadim_popcorn.pkl',
    16: 'models/small_popcorn.pkl'
}

models = {product_id: joblib.load(path) for product_id, path in model_paths.items()}

# ตั้งค่า MongoDB
mongo_client = MongoClient("mongodb+srv://coco:62BkDlEjFthts7s3@cluster0.91fcjua.mongodb.net/?retryWrites=true&w=majority")
db = mongo_client['test']  # ชื่อฐานข้อมูลของคุณคือ 'test'
counts_products_collection = db['counts_products']
import_products_collection = db['import_products']
products_collection = db['products']

# ฟังก์ชั่นเพื่อดึงข้อมูลสภาพอากาศ
def get_weather_data():
    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': 13.91301,
        'lon': 100.49883,
        'appid': '8996956bada63816da0df68bbf3a1c84',
        'units': 'metric'
    }
    response = requests.get(weather_url, params=params)
    data = response.json()

    temp_max = data['main']['temp_max']
    temp_min = data['main']['temp_min']
    temp_avg = (temp_max + temp_min) / 2
    rain = data['rain']['1h'] if 'rain' in data and '1h' in data['rain'] else 0

    return temp_max, temp_min, temp_avg, rain

# ฟังก์ชั่นสำหรับตรวจสอบวันเทศกาล
def is_holiday(date):
    holidays = [
        "01-01", "05-01", "13-01", "15-01", "16-01", "02-02", "04-02", "10-02", "14-02", "24-02",
        "03-03", "08-03", "14-03", "15-03", "21-03", "01-04", "06-04", "07-04", "11-04", "13-04",
        "14-04", "15-04", "22-04", "23-04", "01-05", "04-05", "18-05", "22-05", "31-05", "03-06",
        "05-06", "08-06", "09-06", "21-06", "26-06", "03-07", "06-07", "07-07", "17-07", "20-07",
        "21-07", "28-07", "08-08", "12-08", "13-08", "19-08", "26-08", "01-09", "16-09", "17-09",
        "18-09", "22-09", "30-09", "01-10", "03-10", "07-10", "13-10", "16-10", "23-10", "24-10",
        "31-10", "14-11", "13-11", "15-11", "28-11", "29-11", "01-12", "03-12", "05-12", "10-12",
        "15-12", "25-12", "31-12"
    ]
    date_str = date.strftime("%d-%m")
    return 1 if date_str in holidays else 0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # กำหนดเวลาปัจจุบันเป็นเวลาในประเทศไทย
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz)
        today_str = now.strftime("%Y-%m-%d")

        # ดึงข้อมูลจาก MongoDB
        results = []
        for product_id, model in models.items():
            counts_product_data = counts_products_collection.find_one({"CountDate": today_str, "Product_ID": product_id})
            if counts_product_data is None:
                raise ValueError(f'No data found in counts_products_collection for Product_ID {product_id} on {today_str}')

            import_product_data = import_products_collection.find_one({"Product_ID": product_id})
            if import_product_data is None:
                raise ValueError(f'No data found in import_products_collection for Product_ID {product_id}')

            product_data = products_collection.find_one({"Product_ID": product_id})
            if product_data is None:
                raise ValueError(f'No data found in products_collection for Product_ID {product_id}')

            # ดึงข้อมูลสภาพอากาศ
            temp_max, temp_min, temp_avg, rain = get_weather_data()

            # กำหนดฟีเจอร์ต่างๆ
            features_list = [
                import_product_data['Count'],  # ยอดสินค้ารับเข้า
                counts_product_data['Count_sell'],  # ยอดยกมาขาย
                counts_product_data['expire'],  # หมดอายุ
                temp_max,  # อุณหภูมิสูงสุด (°C)
                temp_min,  # อุณหภูมิต่ำสุด (°C)
                temp_avg,  # อุณหภูมิเฉลี่ย (°C)
                rain,  # ปริมาณน้ำฝนรวม (mm)
                is_holiday(now),  # วันเทศกาล
                1 if now.weekday() == 0 else 0,  # จันทร์
                1 if now.weekday() == 2 else 0,  # พุธ
                1 if now.weekday() == 3 else 0,  # พฤหัสบดี
                1 if now.weekday() == 4 else 0,  # ศุกร์
                1 if now.weekday() == 5 else 0,  # เสาร์
                1 if now.weekday() == 1 else 0,  # อังคาร
                1 if now.weekday() == 6 else 0,  # อาทิตย์
                now.month,  # เดือน
                1 if now.month in [5, 6, 7, 8, 9, 10] else 0,  # ฤดูฝน
                1 if now.month in [3, 4] else 0,  # ฤดูร้อน
                1 if now.month in [11, 12, 1, 2] else 0,  # ฤดูหนาว
            ]

            input_array = np.array(features_list).reshape(1, -1)

            # ตรวจสอบการทำนาย
            print(f'Input array: {input_array} for Product_ID {product_id}')
            prediction = model.predict(input_array)

            # ปัดเศษผลการทำนาย
            rounded_prediction = round(prediction[0])

            # พิมพ์ผลการทำนายลงในคอนโซล
            print(f'Prediction: {rounded_prediction} for Product_ID {product_id}')

            # บันทึกผลการทำนายลงในไฟล์ log
            logging.info(f'Prediction: {rounded_prediction} for Product_ID {product_id}')

            # เพิ่มผลลัพธ์ในรายการ
            results.append({
                'Product_ID': product_id,
                'NameProduct': product_data["NameProduct"],
                'predicted_sales': rounded_prediction
            })

        # ส่งผลการทำนายทั้งหมดกลับไป
        return jsonify(results)

    except Exception as e:
        print(f'Error: {e}')
        logging.error(f'Error: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
