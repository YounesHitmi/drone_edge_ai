import cv2
import numpy as np
import os
import random

NUM_IMAGES = 500
IMG_SIZE = 640
DATASET_DIR = "dataset_formes"
CLASSES = ["circle", "square", "triangle"]

# Création de l'arborescence des dossiers (images et labels)
os.makedirs(f"{DATASET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels/train", exist_ok=True)

print("Image and label generation...")

for i in range(NUM_IMAGES):
    bg_color = np.random.randint(50, 200, 3).tolist()
    img = np.full((IMG_SIZE, IMG_SIZE, 3), bg_color, dtype=np.uint8)

    class_id = random.randint(0, 2)
    color = np.random.randint(0, 255, 3).tolist()
    
    w = random.randint(50, 200)
    h = w if class_id != 2 else random.randint(50, 200) # Carré/circle proportionnels
    x_center = random.randint(w//2, IMG_SIZE - w//2)
    y_center = random.randint(h//2, IMG_SIZE - h//2)
    
    # Calculer les points bruts pour OpenCV
    x1, y1 = int(x_center - w/2), int(y_center - h/2)
    x2, y2 = int(x_center + w/2), int(y_center + h/2)

    if class_id == 0: 
        cv2.circle(img, (x_center, y_center), w//2, color, -1)
    elif class_id == 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    elif class_id == 2: 
        pts = np.array([[x_center, y1], [x1, y2], [x2, y2]], np.int32)
        cv2.fillPoly(img, [pts], color)

    img_path = f"{DATASET_DIR}/images/train/img_{i}.jpg"
    cv2.imwrite(img_path, img)

    label_path = f"{DATASET_DIR}/labels/train/img_{i}.txt"
    with open(label_path, "w") as f:
        x_c_norm = x_center / IMG_SIZE
        y_c_norm = y_center / IMG_SIZE
        w_norm = w / IMG_SIZE
        h_norm = h / IMG_SIZE
        f.write(f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)} 
train: images/train
val: images/train 

nc: 3
names: ['circle', 'square', 'triangle']
"""
with open(f"{DATASET_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print(f"Done. {NUM_IMAGES} images generated in '{DATASET_DIR}' folder.")
print("'data.yaml' is ready for learning")