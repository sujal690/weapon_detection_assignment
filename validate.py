from ultralytics import YOLO

def validate_model(model_path: str, data_yaml: str):
    print(f"Validating model {model_path} on dataset {data_yaml} ...")
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print("\nValidation Metrics:")
    print(f"mAP@0.5: {results.box.map50:.3f}")
    print(f"mAP@0.5-0.95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")

if __name__ == "__main__":
    # Update paths as needed
    model_path = "./yolov8n_results2/weights/best.pt"
    data_yaml = "./Weapon.v3i.yolov8/data.yaml"


    validate_model(model_path, data_yaml)
