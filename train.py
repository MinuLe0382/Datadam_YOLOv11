from ultralytics import YOLO

yaml_file = 'datasets/nail/fingernail.yaml'

# Load a model
model = YOLO("yolo11s-seg.pt")  # load a pretrained model (recommended for training)

results = model.train(data=yaml_file, epochs=100, imgsz=1760, device=[0, 1, 2], batch=21, # imgsz: 32의 배수
                    degrees=15.0,      # 이미지 회전 각도 (±15도)
                    translate=0.1,   # 이미지 이동 비율 (±10%)
                    scale=0.2,       # 이미지 크기 조절/확대 비율 (±20%)
                    fliplr=0.5,      # 50% 확률로 좌우 반전
                    flipud=0.5,      # 50% 확률로 상하 반전
                    hsv_h=0.015,     # 색상(Hue) 변형 강도
                    hsv_s=0.7,       # 채도(Saturation) 변형 강도
                    hsv_v=0.4        # 명도(Value) 변형 강도
)