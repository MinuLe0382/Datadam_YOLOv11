import os
import cv2
import numpy as np
from tqdm import tqdm

mask_dir = "datasets/nail/test/masks"
label_dir = "datasets/nail/test/labels"
class_id = 0 # single class (nail)    

os.makedirs(label_dir, exist_ok=True)
mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"number of masks: {len(mask_files)}")

for mask_filename in tqdm(mask_files, desc="Converting masks to labels"):
    mask_path = os.path.join(mask_dir, mask_filename)
    label_path = os.path.join(label_dir, os.path.splitext(mask_filename)[0] + ".txt")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"경고: {mask_filename} 파일을 읽을 수 없습니다.")
        continue

    h, w = mask.shape

    # find contour
    # cv2.RETR_EXTERNAL: 가장 바깥쪽 외곽선만 찾음
    # cv2.CHAIN_APPROX_SIMPLE: 외곽선의 꼭짓점만 저장하여 용량을 줄임
    # 마스크가 만족스럽지 않으면, cv2.CHAIN_APPROX_SIMPLE 대신 cv2.CHAIN_APPROX_NONE 사용 고려
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_labels = []

    for contour in contours:
        # 너무 작은 객체를 무시
        # if cv2.contourArea(contour) < 5:
            # continue

        # normalize contour points to range [0, 1]
        # contour 배열의 형태를 (N, 2)로 변경 후 (contour는 (N, 1, 2)를 반환. 1은 필요 없는 차원) x, y 각각 너비, 높이로 나눔
        normalized_contour = contour.squeeze(1).astype(np.float32)
        normalized_contour[:, 0] /= w  # x 좌표
        normalized_contour[:, 1] /= h  # y 좌표

        # YOLO 형식 문자열 생성: "class_id x1 y1 x2 y2 ..."
        segment_str = " ".join(map(str, normalized_contour.flatten()))
        yolo_label = f"{class_id} {segment_str}"
        yolo_labels.append(yolo_label)

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_labels))

print("done")