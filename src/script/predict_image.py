
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
from logic import interpret_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "waste_classifier.h5"
CLASS_FILE = BASE_DIR / "models" / "class_names.txt"
IMG_SIZE = (224,224)

def load_classes():
    return [line.strip() for line in open(CLASS_FILE,"r")]

def predict_on_image(path):
    model = keras.models.load_model(str(MODEL_PATH))
    class_names = load_classes()

    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img,0)

    pred = model.predict(img)[0]
    w, cal, carb, route = interpret_prediction(pred, class_names)

    print("Waste:", w)
    print("Calorific:", cal)
    print("Carbon Impact:", carb)
    print("Route:", route)




if __name__=="__main__":
    sample = BASE_DIR.parent / "data" / "garbage" / "plastic" / "plastic91.jpg"
    print("Trying sample:", sample)
    predict_on_image(sample)

