import argparse
from ultralytics import YOLO

def train_yolo(model_name, data_path, epochs=100):
    model_path = "yolov5nu.pt" if model_name == "yolov5nu" else "yolo12n.pt"
    model = YOLO(model_path)
    results = model.train(
        data=data_path,
        epochs=epochs,
        save=True  
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["yolov5nu", "yolo12n"], help="YOLO model to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the YAML dataset file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    train_yolo(args.model_name, args.data_path, args.epochs)
