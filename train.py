from ultralytics import YOLO

# Load a pre-trained YOLO model for segmentation
model = YOLO('yolo11m-seg.pt')  # Consider a larger model for better accuracy
model.yaml = {'nc': 2} 
# Train the model with the updated configuration and hyperparameters
model.train(
    data='config.yaml',  # Path to your configuration file
    epochs=10,           # Set to at least 10 for initial testing
    imgsz=640,           # Image size
    batch=16,            # Batch size
    lr0=0.01,
    momentum=0.937,      
    weight_decay=0.0005,
    warmup_epochs=3,    
    save_period=5,            
)
