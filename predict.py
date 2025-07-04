# predict.py

import torch
import cv2
import numpy as np
from torchvision import transforms
from train import FireCNN  # Assuming same file or import the class

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FireCNN().to(device)
model.load_state_dict(torch.load("fire_cnn.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Real-time camera
cap = cv2.VideoCapture(0)
classes = ["fire", "normal"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    cv2.putText(frame, f"Prediction: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
