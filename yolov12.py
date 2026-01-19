from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO(r"C:\Users\tabao\Downloads\yolo26m-pose (1).pt")

# Run inference on 'bus.jpg' with arguments
model.track(source = r"C:\Users\tabao\Downloads\short2.mp4", show = True)