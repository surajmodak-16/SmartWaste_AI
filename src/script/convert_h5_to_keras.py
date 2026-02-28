from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent.parent
H5_PATH = BASE_DIR / "models" / "waste_classifier.h5"
OUT_PATH = BASE_DIR / "models" / "waste_classifier.keras"
CLASS_FILE = BASE_DIR / "models" / "class_names.txt"

IMG_SIZE = (224, 224)

def build_model(num_classes):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

print("\nReading class names...")
class_names = [l.strip() for l in open(CLASS_FILE, "r") if l.strip()]
num_classes = len(class_names)

print("Building model...")
model = build_model(num_classes)

print("Loading weights from H5 file...")
model.load_weights(str(H5_PATH))

print("Saving clean Keras model (.keras)...")
model.save(str(OUT_PATH))

print("\n✔ DONE! New safe model saved at:")
print(OUT_PATH)
