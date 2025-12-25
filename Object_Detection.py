import cv2
import numpy as np

# Load YOLOv3-Tiny
net = cv2.dnn.readNetFromDarknet(
    "yolov3-tiny.cfg",
    "yolov3-tiny.weights"
)

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open webcam
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape

    # Create blob
    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (416, 416),
        swapRB=True, crop=False
    )

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[class_ids[i]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img, label, (x, y - 10),
                font, 0.6, color, 2
            )

    cv2.imshow("YOLOv3-Tiny Object Detection", img)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
