import os
import pandas as pd
import cv2
import torch
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from utils import detect_image
from detect import process_image, process_video, generate_report
from config import UPLOAD_FOLDER
import csv
import datetime
import matplotlib
import subprocess
from flask_socketio import SocketIO
import time
import os
os.environ["FLASK_RUN_FROM_CLI"] = "false"

REPORTS_DIR = 'static/reports'

def convert_video_ffmpeg(input_path, output_path):
    """Converte vídeo para WebM (VP9) usando FFMPEG"""
    command = [
        "ffmpeg", "-y", "-i", input_path, "-c:v", "libvpx-vp9",
        "-b:v", "2M", "-c:a", "libopus", output_path
    ]
    subprocess.run(command, check=True)


matplotlib.use('Agg')  # Backend não interativo

# Arquivos e configurações
log_file = 'static/reports/detection_log.csv'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
REPORT_FOLDER = 'static/reports' 
LOG_FILE = os.path.join(REPORT_FOLDER, 'detection_log.csv')

# Criar diretórios se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Modelo YOLO
model = YOLO('dataset/runs/detect/train2/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)



# Configuração do Flask

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

socketio = SocketIO(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
progress = {"current": 0}  # Progresso global

# Rota de Página Inicial
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reports/<filename>')
def report(filename):
    # Alinha com a forma como o relatório é salvo
    report_path = os.path.join('static', 'reports', f'{filename}.html')

    if not os.path.exists(report_path):
        return "Relatório não encontrado.", 404

    # ... dados fictícios ou reais, você pode manter
    timestamps = ["10:00", "10:05", "10:10"]
    data = {
        "dormindo": [1, 2, 3],
        "acordado": [4, 5, 6],
        "copiando": [2, 3, 1],
        "atento": [3, 4, 2],
        "distraido": [1, 1, 2]
    }

    return render_template(
        'report.html',
        filename=filename,
        timestamps=timestamps,
        data=data
    )

# Progresso
@app.route('/progress')
def progress_status():
    global progress
    return jsonify(progress)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files: #isso acontece pq eu comi o cu de qm ta lendo
            return jsonify({"error": "Nenhum arquivo foi enviado"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400

        # Salvar arquivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Processar
        if file.filename.endswith(('.jpg', '.jpeg', '.png')):
            detections, processed_path = process_image(file_path)
            # Se for imagem, renderiza diretamente o HTML com gráficos
            return render_template('report.html', graph_path=None, pie_path=None, video_path=processed_path)
        elif file.filename.endswith(('.mp4', '.avi', '.mkv', '.webm')):
            result = process_video(file_path)
            processed_path = result['processed_video']
            filename = os.path.splitext(os.path.basename(processed_path))[0]  # sem replace
            return redirect(url_for('report', filename=filename))  #  Redireciona para HTML dinâmico
        else:
            return jsonify({"error": "Formato de arquivo não suportado"}), 400

    except Exception as e:
        print(f"❌ Erro em /upload: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/history')
def history():
    processed_dir = 'processed'
    files = os.listdir(processed_dir)
    history_data = []
    for filename in files:
        if filename.endswith(('.mp4', '.avi', '.mkv')):
            report_path = url_for('report', filename=os.path.splitext(filename)[0])
            history_data.append({
                "video_path": url_for('processed_file', filename=filename),
                "report_path": report_path,
                "filename": filename,
                "date": "14/12/2024",
                "object_counts": {"Pessoa": 3, "Cadeira": 5}
            })
    return render_template('history.html', history=history_data)



# Dados do Gráfico
@app.route('/graph-data')
def graph_data():
    try:
        print("✅ Iniciando leitura do CSV")
        timestamps = []
        data = {'dormindo': [], 'acordado': [], 'copiando': [], 'atento': [], 'distraido': []}

        with open(LOG_FILE, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            print("📄 Cabeçalhos:", headers)

            for row in reader:
                print("🔹 Linha:", row)
                if len(row) < 6:
                    continue  # Ignora linhas incompletas

                timestamps.append(row[0])
                data['dormindo'].append(int(row[1]))
                data['acordado'].append(int(row[2]))
                data['copiando'].append(int(row[3]))
                data['atento'].append(int(row[4]))
                data['distraido'].append(int(row[5]))

        print("✅ Dados lidos com sucesso")
        return jsonify({"timestamps": timestamps, "data": data})
    except Exception as e:
        print(f"❌ Erro ao carregar /graph-data: {str(e)}")
        return jsonify({"error": str(e)}), 500


# --- FUNÇÕES DE PROCESSAMENTO ---


def enviar_dados_ao_vivo():
    """Envia dados de detecção em tempo real para o gráfico."""
    while True:
        try:
            # Aqui você vai colocar a contagem real do YOLO
            dados = {
                "timestamp": time.strftime("%H:%M:%S"),
                "atento": np.random.randint(0, 10),  # troca depois pelo valor real
                "distraido": np.random.randint(0, 5) # troca depois pelo valor real
            }
            socketio.emit("atualizacao_grafico", dados)
        except Exception as e:
            print(f"❌ Erro no envio em tempo real: {e}")
        socketio.sleep(2)  # envia a cada 2s
        
def process_image(image_path):
    """Processa uma imagem"""
    image = cv2.imread(image_path)
    results = model.predict(image, conf=1)
    detections = {}

    for result in results[0].boxes:
        class_id = int(result.cls[0].item())
        class_name = model.names[class_id]
        detections[class_name] = detections.get(class_name, 0) + 1

    return detections, image_path


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
        results = model.predict(frame, conf=0.1)
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
    """Gera gráficos e cria um relatório HTML usando o template do Flask."""
    try:
        if not detections or all(count == 0 for count in detections.values()):
            print("⚠ Nenhuma detecção encontrada. Nenhum relatório gerado.")
            return {"error": "Nenhuma detecção encontrada."}

        filename = os.path.splitext(os.path.basename(processed_video_path))[0]
        base_name = filename.replace("_processed", "")
        report_path = os.path.join(REPORTS_DIR, f"{base_name}_processed.html")

        # Caminhos dos gráficos
        graph_filename = f"{filename}_graph.png"
        pie_filename = f"{filename}_pie.png"
        graph_path = os.path.join(REPORTS_DIR, graph_filename)
        pie_path = os.path.join(REPORTS_DIR, pie_filename)

        # Geração do gráfico de barras
        plt.figure(figsize=(10, 5))
        plt.bar(detections.keys(), detections.values(), color=['#76c7c0', '#4caf50', '#ff9f40', '#f44336', '#8e44ad'])
        plt.title("Contagem de Objetos Detectados")
        plt.xlabel("Objeto")
        plt.ylabel("Quantidade")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        # Geração do gráfico de pizza
        plt.figure(figsize=(8, 8))
        plt.pie(detections.values(), labels=detections.keys(), autopct='%1.1f%%',
                colors=['#76c7c0', '#4caf50', '#ff9f40', '#f44336', '#8e44ad'],
                startangle=140, wedgeprops={'edgecolor': 'black'})
        plt.title("Distribuição de Objetos Detectados")
        plt.savefig(pie_path)
        plt.close()

        # Renderiza HTML
        html_content = render_template(
            'report.html',
            filename=base_name,
            graph_path=f"/static/reports/{graph_filename}",
            pie_path=f"/static/reports/{pie_filename}",
            video_path=f"/{processed_video_path}"
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Relatório HTML gerado em: {report_path}")
        return {
            "processed_video": processed_video_path,
            "report": report_path
        }

    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {str(e)}")
        return {"error": str(e)}

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
    results = model.predict(image, conf=0.35)
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

    return detections, output_path

# --- EXECUÇÃO DO FLASK ---
if __name__ == '__main__':
    socketio.start_background_task(enviar_dados_ao_vivo)
    socketio.run(app, host="localhost", port=5000, debug=True)


def live_detection(selected_camera_index=0):
    cap = cv2.VideoCapture(selected_camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def select_camera():
    camera_index = simpledialog.askinteger("Selecionar Câmera", "Digite o índice da câmera (0 para a primeira câmera, 1 para a segunda, etc.):")
    return camera_index

# Configuração da interface gráfica
root = tk.Tk()
root.title("Reconhecimento ao vivo")
root.geometry("600x400")
root.configure(background="#f0f0f0")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 14), background="#f0f0f0")

title_label = ttk.Label(root, text="Reconhecimento ao vivo")
title_label.pack(pady=20)

btn_live_detection = ttk.Button(root, text="Detecção ao Vivo", command=lambda: live_detection(select_camera()))
btn_live_detection.pack(pady=10)


root.mainloop()
