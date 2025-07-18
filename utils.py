import cv2
from ultralytics import YOLO

# Carregue o modelo YOLO uma vez
model = YOLO('dataset/runs/detect/train2/weights/best.pt')

def detect_image(image):
    print("Iniciando detecção de imagem...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, conf=0.25)

    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        class_id = int(result.cls[0].item())
        confidence = result.conf[0].item()
        class_name = model.names[class_id]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image
