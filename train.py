from ultralytics import YOLO

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train on our drone dataset
model.train(
    data="drone_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="aeroguard_drone"
)