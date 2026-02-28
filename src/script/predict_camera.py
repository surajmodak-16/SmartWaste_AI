
import cv2, numpy as np
from tensorflow import keras
from pathlib import Path
from logic import interpret_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "waste_classifier.keras"
CLASS_FILE = BASE_DIR / "models" / "class_names.txt"
IMG_SIZE = (224,224)

def load_classes():
    return [l.strip() for l in open(CLASS_FILE)]

def main():
    model = keras.models.load_model(str(MODEL_PATH))
    classes = load_classes()

    cap = cv2.VideoCapture(0)
    print("Press C to classify, Q to quit")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)&0xFF

        if key==ord('c'):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img,0)
            pred = model.predict(img)[0]

            w,cal,carb,route = interpret_prediction(pred, classes)
            print("Detected:", w, "| Cal:", cal, "| Carbon:", carb, "| Route:", route)

        if key==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
