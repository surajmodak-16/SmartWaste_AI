import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # smart-waste-ai/src
sys.path.append(str(BASE_DIR))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware   # ⬅️ ADD THIS
from tensorflow import keras
import numpy as np
import cv2
import datetime
from logic import interpret_prediction
from pymongo import MongoClient

MODEL_PATH = BASE_DIR.parent / "models" / "waste_classifier.keras"
CLASS_FILE = BASE_DIR.parent / "models" / "class_names.txt"
IMG_SIZE = (224, 224)

app = FastAPI(title="Smart Waste AI API")

# ⬇️ ALLOW REACT (VITE) FRONTEND TO CALL THIS API
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_classes():
    return [line.strip() for line in open(CLASS_FILE, "r")]

model = keras.models.load_model(str(MODEL_PATH))
classes = load_classes()

client = MongoClient("mongodb://localhost:27017/")
db = client["smart_waste_db"]
collection = db["waste_records"]


from bson import ObjectId

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    data = await file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, 0)

    pred = model.predict(img)[0]
    w, cal, carb, route = interpret_prediction(pred, classes)

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "waste_type": w,
        "calorific_value": cal,
        "carbon_impact": carb,
        "route": route,
        "image_name": file.filename,
    }

    inserted = collection.insert_one(record)

    # Convert MongoDB ObjectId to string properly
    record["_id"] = str(inserted.inserted_id)

    return record



@app.get("/records")
def get_records(limit: int = 50):
    logs = list(collection.find().sort("timestamp", -1).limit(limit))
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs


@app.get("/")
def root():
    return {"message": "Smart AI Waste Classifier API is running!"}



@app.get("/test-db")
def test_db():
    try:
        db.list_collection_names()
        return {"message": "MongoDB Connected Successfully!"}
    except Exception as e:
        return {"error": str(e)}

