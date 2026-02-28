
import numpy as np

CALORIFIC = {
    "cardboard": 3.5,
    "glass": 0.2,
    "metal": 1.0,
    "paper": 3.0,
    "plastic": 4.5,
    "trash": 2.0,
}

CARBON_IMPACT = {
    "cardboard": "Medium",
    "glass": "Low",
    "metal": "Medium",
    "paper": "Medium",
    "plastic": "High",
    "trash": "High",
}

DISPOSAL_ROUTE = {
    "cardboard": "Recycling",
    "glass": "Recycling",
    "metal": "Recycling",
    "paper": "Recycling",
    "plastic": "Recycling / Energy Recovery",
    "trash": "Landfill / Special Processing",
}

def interpret_prediction(pred, class_names):
    idx = int(np.argmax(pred))
    w = class_names[idx]
    return w, CALORIFIC.get(w), CARBON_IMPACT.get(w), DISPOSAL_ROUTE.get(w)
