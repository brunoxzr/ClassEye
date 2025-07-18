import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import UPLOAD_FOLDER
from utils import detect_image
import datetime
import csv
import subprocess
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
REPORT_FOLDER = 'static/reports'
REPORTS_DIR = 'static/reports'
LOG_FILE = os.path.join(REPORT_FOLDER, 'detection_log.csv')
def convert_video_ffmpeg(input_path, output_path):
    """Converte vídeo para WebM (VP9) usando FFMPEG"""
    command = [
        "ffmpeg", "-y", "-i", input_path, "-c:v", "libvpx-vp9",
        "-b:v", "2M", "-c:a", "libopus", output_path
    ]
    subprocess.run(command, check=True)

log_file = 'static/reports/detection_log.csv'


# Criar arquivo CSV 
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Dormindo', 'Acordado', 'Copiando', 'Atento', 'Distraido'])

# Modelo do yolo
model = YOLO('dataset/runs/detect/train2/weights/best.pt')
def log_detection(counts):
    """Registra os números de detecção no CSV."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            current_time,
            counts.get('dormindo', 0),
            counts.get('acordado', 0),
            counts.get('copiando', 0),
            counts.get('atento', 0),
            counts.get('distraido', 0)
        ])
    print(f"✅ Log atualizado: {LOG_FILE}")

def process_video(video_path):
    """Processa o vídeo e gera os relatórios."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Garante que o nome do arquivo está correto
    filename = os.path.basename(video_path).replace(".mp4", "") + "_processed.mp4"
    processed_dir = "processed"
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, filename)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detections_count = {"dormindo": 0, "acordado": 0, "copiando": 0, "atento": 0, "distraido": 0}
    
    frame_count = 0
    frame_skip = 5  # Pular frames para otimizar processamento

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Pula frames para reduzir carga

        # Inferência YOLO
        results = model.predict(frame, conf=0.5)
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            class_name = model.names[class_id]

            if class_name in detections_count:
                detections_count[class_name] += 1

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    print(f"✅ Vídeo processado salvo em: {output_path}")
    log_detection(detections_count)
    return generate_report(detections_count, output_path)

def generate_report(detections, processed_video_path):
    """
    Após processar o vídeo, apenas retorna o caminho do vídeo processado.
    Os gráficos serão gerados dinamicamente via /graph-data no report.html.
    """
    if not detections or all(count == 0 for count in detections.values()):
        print("⚠ Nenhuma detecção encontrada.")
        return {"error": "Nenhuma detecção encontrada."}

    print("✅ Relatório pronto para visualização dinâmica.")
    return {
        "processed_video": processed_video_path
    }

def process_image(image_path):
    global progress
    progress["current"] = 0

    # Lê a imagem
    image = cv2.imread(image_path)
    progress["current"] = 20

    # Verifica se a imagem foi carregada corretamente
    if image is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")

    # Realiza inferência com o modelo YOLO
    results = model.predict(image, conf=0.1)
    progress["current"] = 70

    detections = {}

    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        confidence = result.conf[0].item()
        class_id = int(result.cls[0].item())
        class_name = model.names[class_id]

        detections[class_name] = detections.get(class_name, 0) + 1

        label = f"{class_name} {confidence:.2f}"
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    output_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    cv2.imwrite(output_path, image)
    progress["current"] = 100
