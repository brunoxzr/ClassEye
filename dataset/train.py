from ultralytics import YOLO

def main():
    model = YOLO('yolo11n.pt')  # ou yolov8m.pt se estiver com problema de memória
    model.train(
        data='dataset.yaml',
        epochs=200,
        imgsz=960,
        batch=5,
        device=0,
        workers=5  # ou 0 se continuar dando erro
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
