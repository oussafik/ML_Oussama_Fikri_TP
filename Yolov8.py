from ultralytics import YOLO
import cv2

model_path = "Yolov8Model.pt"
image_path = "img.jpg"

img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)

keypoints = results[0].keypoints.xy  # Access the xy attribute of the Keypoints object

# Convert the image to BGR color format
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

for keypoint_index in range(keypoints.shape[1]):
    x, y = int(keypoints[0, keypoint_index, 0]), int(keypoints[0, keypoint_index, 1])
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(img, str(keypoint_index), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)