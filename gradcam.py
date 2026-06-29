"""
gradcam.py — Task 5: Explainable AI
=======================================
Implements Grad-CAM for the EfficientNet-B0 backbone.
Targets the final convolutional layer to visualize model attention.
"""

import cv2
import torch
import numpy as np
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        """
        Computes the Grad-CAM heatmap for the given input tensor.
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        # Compute weights (Global Average Pooling on gradients)
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        # Resize to match input image
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        return cam

def overlay_cam_on_image(img_pil, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap on the original PIL image.
    """
    img_np = np.array(img_pil)
    
    # Ensure image is RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay

def generate_gradcam_overlay(model, input_tensor, original_image, target_layer=None):
    """
    Helper to run Grad-CAM efficiently in the Streamlit app.
    For EfficientNet-B0, target_layer is typically model.features[-1]
    """
    if target_layer is None:
        try:
            # PyTorch EfficientNet architecture
            target_layer = model.features[-1]
        except AttributeError:
            raise ValueError("Could not automatically determine target layer. Please provide it.")
            
    gcam = GradCAM(model, target_layer)
    cam = gcam(input_tensor)
    overlay = overlay_cam_on_image(original_image, cam)
    return overlay, cam
