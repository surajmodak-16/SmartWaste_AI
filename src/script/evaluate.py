# src/evaluate.py
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import itertools, matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "garbage"
MODEL = BASE / "models" / "waste_classifier.keras"
CLASS_FILE = BASE / "models" / "class_names.txt"
IMG_SIZE = (224,224)
BATCH = 32

model = keras.models.load_model(str(MODEL))
with open(CLASS_FILE,'r') as f:
    classes = [l.strip() for l in f if l.strip()]

ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(DATA_DIR),
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

y_true = []
y_pred = []
for imgs, labels in ds:
    preds = model.predict(imgs)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(preds,axis=1).tolist())

print(classification_report(y_true, y_pred, target_names=classes))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\\n", cm)
# Optional: plot matrix
plt.imshow(cm, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
plt.tight_layout()
plt.show()
