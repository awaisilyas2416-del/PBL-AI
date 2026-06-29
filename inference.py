"""
inference.py — End-to-end Inference Pipeline
==============================================
Chains Segmentation (ROI Extraction) -> Classification -> Grad-CAM.
Useful for deploying in Streamlit and running tests.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

from monai.networks.nets import UNet
from monai.transforms import LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, EnsureTyped, Compose
from torchvision.transforms import functional as F_vision

from classification_train import build_model
from config import (
    SEG_MODEL_PATH, CLASS_MODEL_PATH, SEG_IMAGE_SIZE, CLASS_IMAGE_SIZE,
    CLASS_NAMES, CLASS_EVAL_TRANSFORM, get_device
)

class MedicalPipeline:
    def __init__(self):
        self.device = get_device()
        self.seg_model = None
        self.class_model = None
        self._load_models()
        
    def _load_models(self):
        """Loads both segmentation and classification weights if they exist."""
        # 1. Seg Model
        self.seg_model = UNet(
            spatial_dims=2, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
        ).to(self.device)
        
        if os.path.exists(SEG_MODEL_PATH):
            self.seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=self.device))
        self.seg_model.eval()
        
        # 2. Class Model
        self.class_model = build_model(self.device)
        if os.path.exists(CLASS_MODEL_PATH):
            self.class_model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=self.device))
        self.class_model.eval()
        
        # 3. Seg transforms (inference version requires a dict input since MONAI is dict-based)
        self.seg_transform = Compose([
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=SEG_IMAGE_SIZE),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ])
        
    def segment_image(self, img_path):
        """Runs U-Net to extract Kidney ROI Mask."""
        # Load grayscale for MONAI
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            raise ValueError(f"Could not read {img_path}")
            
        img_dict = {"image": np.expand_dims(img_np, axis=-1)} # add channel dim for MONAI
        img_tensor = self.seg_transform(img_dict)["image"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask_out = self.seg_model(img_tensor)
            mask_pred = (torch.sigmoid(mask_out) > 0.5).float().cpu().numpy()[0, 0]
            
        # Resize mask back to original image size
        mask_pred = cv2.resize(mask_pred, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_pred
        
    def extract_roi(self, img_path, mask):
        """Crops the original image using the segmentation mask bounding box."""
        original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Find contours of the mask to get the bounding box
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return Image.fromarray(original_img) # Fallback if no mask found
            
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Add padding to ROI
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original_img.shape[1] - x, w + 2*padding)
        h = min(original_img.shape[0] - y, h + 2*padding)
        
        roi_crop = original_img[y:y+h, x:x+w]
        return Image.fromarray(roi_crop)
        
    def classify_roi(self, roi_pil):
        """Runs EfficientNet-B0 on the extracted ROI."""
        tensor = CLASS_EVAL_TRANSFORM(roi_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.class_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        pred_class = CLASS_NAMES[pred_idx]
        
        return {
            "predicted_class": pred_class,
            "confidence": float(confidence),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        }
        
    def predict_end_to_end(self, img_path):
        """Full pipeline execution."""
        mask = self.segment_image(img_path)
        roi_pil = self.extract_roi(img_path, mask)
        class_results = self.classify_roi(roi_pil)
        
        class_results["mask"] = mask
        class_results["roi_image"] = roi_pil
        return class_results

# Example usage
if __name__ == "__main__":
    pipeline = MedicalPipeline()
    print("Pipeline loaded successfully.")
