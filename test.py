import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

MODEL_PATH = 'runs/segment/train10/weights/best.pt'
TEST_IMAGE_DIR = 'datasets/nail/val/images'
GROUND_TRUTH_MASK_DIR = 'datasets/nail/val/masks'
VISUALIZE_IMAGE_INDEX = -1

def calculate_iou(pred_mask, gt_mask):
    pred_mask_bool = pred_mask.astype(bool)
    gt_mask_bool = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union = np.logical_or(pred_mask_bool, gt_mask_bool).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def main():
    run_dir = os.path.dirname(os.path.dirname(MODEL_PATH)) 
    run_name = os.path.basename(run_dir) 

    base_save_dir = os.path.join('test_results', run_name)
    masks_save_dir = os.path.join(base_save_dir, 'masks')
    viz_save_dir = os.path.join(base_save_dir, 'visualizations')
    os.makedirs(masks_save_dir, exist_ok=True)
    os.makedirs(viz_save_dir, exist_ok=True)
    print(f"Results will be saved in: {base_save_dir}")

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"오류: '{TEST_IMAGE_DIR}' 폴더에 이미지가 없습니다.")
        return

    total_iou = 0
    num_images_with_masks = 0

    print(f"Found {len(image_files)} test images. Starting evaluation...")
    for i, filename in enumerate(tqdm(image_files, desc="Processing test images")):
        image_path = os.path.join(TEST_IMAGE_DIR, filename)
        gt_mask_path = os.path.join(GROUND_TRUTH_MASK_DIR, os.path.splitext(filename)[0] + '.png')

        results = model.predict(image_path, verbose=False, conf=0.1)
        result = results[0]
        original_image = result.orig_img

        pred_mask_combined = np.zeros(result.orig_shape, dtype=np.uint8)
        if result.masks is not None:
            orig_h, orig_w = result.orig_shape
            for mask_tensor in result.masks.data:
                single_mask = mask_tensor.cpu().numpy().astype(np.uint8)
                single_mask_resized = cv2.resize(single_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                pred_mask_combined = np.maximum(pred_mask_combined, single_mask_resized)

        mask_save_path = os.path.join(masks_save_dir, filename)
        cv2.imwrite(mask_save_path, pred_mask_combined * 255)

        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            iou = calculate_iou(pred_mask_combined, gt_mask)
            total_iou += iou
            num_images_with_masks += 1
        else:
            iou = -1

        if i == VISUALIZE_IMAGE_INDEX:
            print(f"\n--- Visualizing image {i}: {filename} ---")
            annotated_frame = result.plot()
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.title('Prediction Overlay')
            plt.axis('off')
            if os.path.exists(gt_mask_path):
                plt.subplot(1, 3, 3)
                plt.imshow(gt_mask, cmap='gray')
                plt.title(f'Ground Truth Mask\nIoU for this image: {iou:.4f}')
                plt.axis('off')
            
            viz_save_path = os.path.join(viz_save_dir, f'visualization_{filename}')
            plt.savefig(viz_save_path)
            print(f"Visualization saved to {viz_save_path}")
    
    if num_images_with_masks > 0:
        mean_iou = total_iou / num_images_with_masks
        print(f"\n--- Evaluation Complete ---")
        print(f"mIoU over {num_images_with_masks} images: {mean_iou:.4f}")
    else:
        print("\n--- Evaluation Complete ---")
        print("정답 마스크가 없어 mIoU를 계산할 수 없습니다.")

if __name__ == '__main__':
    main()
