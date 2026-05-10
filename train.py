from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print("Training...")
results = model.train(
    data='dataset_formes/data.yaml', 
    epochs=15,                       
    imgsz=640,                      
    batch=16,                       
    device='cpu',                   
    project='runs',                 
    name='entrainement_formes'       
)

print("Training done. Model Saved.")