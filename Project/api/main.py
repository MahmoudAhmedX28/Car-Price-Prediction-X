import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

with open('../models/one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('../models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../models/model.pkl', 'rb') as f:
    model = pickle.load(f)


class CarInput(BaseModel):
    Levy: int
    Manufacturer: str
    Model: str
    Prod_year: int
    Category: str
    Leather_interior: str
    Fuel_type: str
    Engine_volume: float
    Mileage: int
    Cylinders: float
    Gear_box_type: str
    Drive_wheels: str
    Wheel: str
    Color: str
    Airbags: int


@app.post("/predict/")
def predict(car_data: CarInput):
    try:
        data = pd.DataFrame([car_data.model_dump()])

        data['Age'] = datetime.now().year - data['Prod_year']
        data = data.drop(columns=['Doors', 'Prod_year'], errors='ignore')

        data.drop(columns=['Prod_year'],errors='ignore',inplace=True)

        column_rename_map ={
            'Leather_interior': 'Leather interior',
            'Gear_box_type': 'Gear box type',
            'Drive_wheels': 'Drive wheels',
            'Engine_volume': 'Engine volume',
            'Fuel_type': 'Fuel type',
        }

        data.rename(columns=column_rename_map, inplace=True)
        one_hot_columns = ['Leather interior', 'Gear box type', 'Drive wheels', 'Wheel']
        encoded_data = one_hot_encoder.transform(data[one_hot_columns])
        encoded_data_df = pd.DataFrame(encoded_data, 
                columns=one_hot_encoder.get_feature_names_out(one_hot_columns),
                index=data.index)

        data = pd.concat([data, encoded_data_df], axis=1)
        data.drop(columns=one_hot_columns, inplace=True)

        label_encoded_columns = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Color']
        for column in label_encoded_columns:
            if column in data.columns:
                le = label_encoders[column]
                data[column] = le.transform(data[column])

        numerical_columns = ['Levy', 'Engine volume', 'Mileage', 'Age']
        data[numerical_columns] = scaler.transform(data[numerical_columns])

        predictions = model.predict(data)
        return {"predictions": predictions[0]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


