"""
Annotation Module
=================
Vision-language annotation using Grounding DINO for object detection.
Supports multiple annotation modes: Grounding DINO, BLIP-2, and basic bounding boxes.
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# ==================== CONFIGURATION ====================
class AnnotationConfig:
    """Configuration for annotation pipeline"""
    
    # Grounding DINO paths - using correct path
    GROUNDING_DINO_PATH = '/home/robot/Documents/VM_Annotation_Pipeline/Grounded-SAM-2'
    GDINO_CONFIG = os.path.join(
        GROUNDING_DINO_PATH,
        "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    )
    GDINO_CHECKPOINT = os.path.join(
        GROUNDING_DINO_PATH,
        "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
    )
    
    # Allowed classes file
    ALLOWED_CLASSES_FILE = os.path.join(
        GROUNDING_DINO_PATH,
        "annotation_pipeline/label_class.txt"
    )
    
    # Output directory
    OUTPUT_BASE_DIR = "pipeline_outputs/annotations"
    
    # Detection thresholds
    BOX_THRESHOLD = 0.20
    TEXT_THRESHOLD = 0.25
    MIN_CONFIDENCE = 0.25
    
    # NMS parameters
    ENABLE_NMS = True
    NMS_IOU_THRESHOLD = 0.5
    NMS_SCORE_THRESHOLD = 0.25
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Default allowed objects
    DEFAULT_ALLOWED_OBJECTS = {
        "traffic_cone", "fire_extinguisher", "cardbox", "pallet",
        "bottle", "vase", "chair", "monitor", "plastic_box",
        "shoe", "apple", "robotic_arm", "cup", "screw_driver",
        "plier", "hammer"
    }

# ==================== UTILITY FUNCTIONS ====================
def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    box1_area = max(0, (x1_max - x1_min)) * max(0, (y1_max - y1_min))
    box2_area = max(0, (x2_max - x2_min)) * max(0, (y2_max - y2_min))
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def nms_per_class(
    detections: List[Dict],
    iou_threshold: float,
    score_threshold: float
) -> List[Dict]:
    """Apply Non-Maximum Suppression per class"""
    if not detections:
        return []
    
    # Filter by score threshold
    detections = [d for d in detections if d['confidence'] >= score_threshold]
    if not detections:
        return []
    
    # Group by class
    detections_by_class = defaultdict(list)
    for det in detections:
        detections_by_class[det['class_id']].append(det)
    
    kept = []
    
    for class_id, class_dets in detections_by_class.items():
        # Sort by confidence descending
        class_dets = sorted(class_dets, key=lambda d: d['confidence'], reverse=True)
        
        suppressed = set()
        for i, det_i in enumerate(class_dets):
            if i in suppressed:
                continue
            kept.append(det_i)
            
            for j in range(i + 1, len(class_dets)):
                if j in suppressed:
                    continue
                iou = compute_iou(det_i['bbox_xyxy'], class_dets[j]['bbox_xyxy'])
                if iou >= iou_threshold:
                    suppressed.add(j)
    
    # Reassign detection IDs
    for idx, det in enumerate(kept, start=1):
        det['detection_id'] = idx
    
    return kept

def cxcywh_norm_to_xyxy(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int
) -> List[int]:
    """Convert normalized center format to absolute xyxy"""
    x_min = int((cx - w / 2) * img_w)
    y_min = int((cy - h / 2) * img_h)
    x_max = int((cx + w / 2) * img_w)
    y_max = int((cy + h / 2) * img_h)
    
    x_min = max(0, min(img_w - 1, x_min))
    y_min = max(0, min(img_h - 1, y_min))
    x_max = max(0, min(img_w - 1, x_max))
    y_max = max(0, min(img_h - 1, y_max))
    
    return [x_min, y_min, x_max, y_max]

def xyxy_to_yolo_normalized(
    box: List[int], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert xyxy to YOLO normalized format (cx, cy, w, h in [0,1])"""
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    cx_norm = max(0, min(1, cx / img_w))
    cy_norm = max(0, min(1, cy / img_h))
    w_norm = max(0, min(1, w / img_w))
    h_norm = max(0, min(1, h / img_h))
    
    return cx_norm, cy_norm, w_norm, h_norm

# ==================== OBJECT NAME MAPPINGS ====================
class ObjectMapper:
    """Maps detected object names to allowed classes"""
    
    @staticmethod
    def get_mappings() -> Dict[str, str]:
        """Comprehensive object name mappings"""
        return {
            # Traffic cone
            "traffic_cone": "traffic_cone",
            "traffic cone": "traffic_cone",
            "cone": "traffic_cone",
            "safety cone": "traffic_cone",
            "road cone": "traffic_cone",
            "orange cone": "traffic_cone",
            
            # Fire extinguisher
            "fire_extinguisher": "fire_extinguisher",
            "fire extinguisher": "fire_extinguisher",
            "extinguisher": "fire_extinguisher",
            
            # Cardbox
            "cardbox": "cardbox",
            "cardboard": "cardbox",
            "cardboard box": "cardbox",
            "box": "cardbox",
            "carton": "cardbox",
            
            # Pallet
            "pallet": "pallet",
            "wooden_pallet": "pallet",
            "wooden pallet": "pallet",
            "wood pallet": "pallet",
            
            # Bottle
            "bottle": "bottle",
            "water bottle": "bottle",
            "plastic bottle": "bottle",
            
            # Vase
            "vase": "vase",
            "flower vase": "vase",
            
            # Chair
            "chair": "chair",
            "seat": "chair",
            
            # Monitor
            "monitor": "monitor",
            "screen": "monitor",
            "display": "monitor",
            "computer monitor": "monitor",
            
            # Plastic box
            "plastic_box": "plastic_box",
            "plastic box": "plastic_box",
            "storage box": "plastic_box",
            "container": "plastic_box",
            
            # Shoe
            "shoe": "shoe",
            "shoes": "shoe",
            "footwear": "shoe",
            
            # Apple
            "apple": "apple",
            "fruit": "apple",
            
            # Robotic arm
            "robotic_arm": "robotic_arm",
            "robotic arm": "robotic_arm",
            "robot arm": "robotic_arm",
            "robot": "robotic_arm",
            "manipulator": "robotic_arm",
            
            # Cup
            "cup": "cup",
            "mug": "cup",
            "glass": "cup",
            
            # Screw driver
            "screw_driver": "screw_driver",
            "screwdriver": "screw_driver",
            "screw driver": "screw_driver",
            
            # Plier
            "plier": "plier",
            "pliers": "plier",
            
            # Hammer
            "hammer": "hammer",
            "tool": "hammer",
        }
    
    @staticmethod
    def normalize(obj_name: str) -> str:
        """Normalize object name using mapping rules"""
        mappings = ObjectMapper.get_mappings()
        obj_clean = obj_name.strip().lower()
        obj_normalized = obj_clean.replace("-", "_").replace(" ", "_")
        
        # Direct match
        if obj_normalized in mappings:
            return mappings[obj_normalized]
        
        # Try with spaces
        obj_spaces = obj_normalized.replace("_", " ")
        if obj_spaces in mappings:
            return mappings[obj_spaces]
        
        # Try without separators
        obj_no_sep = obj_clean.replace("-", "").replace("_", "").replace(" ", "")
        for key in mappings:
            key_no_sep = key.replace("-", "").replace("_", "").replace(" ", "")
            if obj_no_sep == key_no_sep:
                return mappings[key]
        
        # Partial match
        for key, value in mappings.items():
            if obj_clean in key or key in obj_clean:
                return value
        
        return obj_normalized

# ==================== CLASS MANAGER ====================
class AllowedClassManager:
    """Manages allowed object classes"""
    
    def __init__(self, class_file: Optional[str] = None):
        self.class_file = class_file
        self.class_id_to_name: Dict[int, str] = {}
        self.class_name_to_id: Dict[str, int] = {}
        self.class_names_list: List[str] = []
        self.name_mappings = ObjectMapper.get_mappings()
        
        if class_file and os.path.exists(class_file):
            self._load_from_file()
        else:
            self._load_default_classes()
    
    def _load_from_file(self):
        """Load classes from label_class.txt"""
        with open(self.class_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            class_id = int(parts[0].strip())
            class_name = parts[1].strip().lower()
            
            self.class_id_to_name[class_id] = class_name
            self.class_name_to_id[class_name] = class_id
            self.class_names_list.append(class_name)
        
        print(f"Loaded {len(self.class_id_to_name)} allowed classes from file")
    
    def _load_default_classes(self):
        """Load default classes"""
        config = AnnotationConfig()
        for idx, class_name in enumerate(sorted(config.DEFAULT_ALLOWED_OBJECTS)):
            self.class_id_to_name[idx] = class_name
            self.class_name_to_id[class_name] = idx
            self.class_names_list.append(class_name)
        
        print(f"Loaded {len(self.class_id_to_name)} default allowed classes")
    
    def match_phrase_to_class(self, phrase: str) -> Optional[Tuple[int, str]]:
        """Match detected phrase to allowed class"""
        phrase_normalized = ObjectMapper.normalize(phrase)
        
        # Direct match
        if phrase_normalized in self.class_name_to_id:
            cid = self.class_name_to_id[phrase_normalized]
            return cid, self.class_id_to_name[cid]
        
        # Try variations
        variations = [
            phrase_normalized.replace('_', ' '),
            phrase_normalized.replace('_', ''),
            phrase_normalized.replace(' ', '_'),
            phrase_normalized.replace(' ', ''),
        ]
        
        for variation in variations:
            if variation in self.class_name_to_id:
                cid = self.class_name_to_id[variation]
                return cid, self.class_id_to_name[cid]
        
        # Partial match
        for class_name, class_id in self.class_name_to_id.items():
            class_variations = [
                class_name,
                class_name.replace('_', ' '),
                class_name.replace('_', ''),
                class_name.replace(' ', '_')
            ]
            for variation in class_variations:
                if variation in phrase_normalized or phrase_normalized in variation:
                    return class_id, class_name
        
        return None

# ==================== GROUNDING DINO DETECTOR ====================
class GroundingDINODetector:
    """Wrapper for Grounding DINO inference"""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self._setup_environment()
        self._load_model()
    
    def _setup_environment(self):
        """Setup Python path for Grounding DINO"""
        if self.config.GROUNDING_DINO_PATH not in sys.path:
            sys.path.insert(0, self.config.GROUNDING_DINO_PATH)
    
    def _load_model(self):
        """Load Grounding DINO model"""
        print(f"Loading Grounding DINO on {self.config.DEVICE}...")
        
        try:
            from grounding_dino.groundingdino.util.inference import load_model
            
            self.model = load_model(
                model_config_path=self.config.GDINO_CONFIG,
                model_checkpoint_path=self.config.GDINO_CHECKPOINT
            )
            print("✓ Grounding DINO model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Grounding DINO: {e}")
    
    def detect(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ):
        """Run detection with given caption"""
        from grounding_dino.groundingdino.util.inference import predict
        
        box_thr = box_threshold if box_threshold is not None else self.config.BOX_THRESHOLD
        text_thr = text_threshold if text_threshold is not None else self.config.TEXT_THRESHOLD
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.config.DEVICE)
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=tensor,
            caption=caption,
            box_threshold=box_thr,
            text_threshold=text_thr,
            device=self.config.DEVICE
        )
        
        return boxes, logits, phrases

# ==================== ANNOTATION ENGINE ====================
class AnnotationEngine:
    """Main annotation engine"""
    
    def __init__(self):
        self.config = AnnotationConfig()
        os.makedirs(self.config.OUTPUT_BASE_DIR, exist_ok=True)
        
        self.class_manager = AllowedClassManager(self.config.ALLOWED_CLASSES_FILE)
        self.detector = None  # Lazy loading
    
    def _ensure_detector(self):
        """Ensure detector is loaded (lazy loading)"""
        if self.detector is None:
            self.detector = GroundingDINODetector(self.config)
    
    def _draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox_xyxy']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return annotated
    
    def annotate_with_grounding_dino(
        self,
        image_path: str,
        prompt: str
    ) -> Dict:
        """Annotate image using Grounding DINO"""
        self._ensure_detector()
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        img_h, img_w = image.shape[:2]
        
        # Run detection
        boxes, logits, phrases = self.detector.detect(
            image=image,
            caption=prompt
        )
        
        # Process detections
        detections = []
        detection_id = 0
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            confidence = float(logit)
            if confidence < self.config.MIN_CONFIDENCE:
                continue
            
            # Match to allowed class
            class_match = self.class_manager.match_phrase_to_class(phrase)
            if class_match is None:
                continue
            
            class_id, class_name = class_match
            
            # Convert box format
            xyxy = cxcywh_norm_to_xyxy(*box.tolist(), img_w, img_h)
            cx_norm, cy_norm, w_norm, h_norm = xyxy_to_yolo_normalized(xyxy, img_w, img_h)
            
            detection_id += 1
            detections.append({
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": class_name,
                "phrase_detected": phrase,
                "confidence": confidence,
                "bbox_xyxy": xyxy,
                "bbox_normalized_cxcywh": [cx_norm, cy_norm, w_norm, h_norm],
                "bbox_yolo_format": f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}",
            })
        
        # Apply NMS
        if self.config.ENABLE_NMS and detections:
            detections = nms_per_class(
                detections,
                self.config.NMS_IOU_THRESHOLD,
                self.config.NMS_SCORE_THRESHOLD
            )
        
        return detections
    
    def annotate_image(
        self,
        image_path: str,
        mode: str = "grounding_dino",
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Annotate image with specified mode
        
        Args:
            image_path: Path to image
            mode: "grounding_dino", "blip2", or "basic"
            prompt: Detection prompt (required for grounding_dino)
        
        Returns:
            Dictionary with annotation results
        """
        print(f"Annotating {os.path.basename(image_path)} with mode: {mode}")
        
        # For now, only grounding_dino is fully implemented
        if mode != "grounding_dino":
            raise NotImplementedError(f"Mode '{mode}' not yet implemented")
        
        if not prompt:
            # Use default objects
            prompt = ", ".join(self.class_manager.class_names_list)
        
        # Run detection
        detections = self.annotate_with_grounding_dino(image_path, prompt)
        
        # Read image for visualization
        image = cv2.imread(image_path)
        
        # Draw annotations
        annotated_image = self._draw_annotations(image, detections)
        
        # Save annotated image
        base_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        annotated_image_path = os.path.join(
            self.config.OUTPUT_BASE_DIR,
            f"{base_name}_annotated_{timestamp}.jpg"
        )
        cv2.imwrite(annotated_image_path, annotated_image)
        
        # Save YOLO annotation file
        annotation_file_path = os.path.join(
            self.config.OUTPUT_BASE_DIR,
            f"{base_name}_{timestamp}.txt"
        )
        with open(annotation_file_path, 'w') as f:
            for det in detections:
                f.write(det['bbox_yolo_format'] + '\n')
        
        # Save JSON metadata
        metadata_path = os.path.join(
            self.config.OUTPUT_BASE_DIR,
            f"{base_name}_metadata_{timestamp}.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump({
                "source_image": image_path,
                "mode": mode,
                "prompt": prompt,
                "num_detections": len(detections),
                "detections": detections,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✓ Found {len(detections)} objects")
        print(f"✓ Saved annotated image: {annotated_image_path}")
        print(f"✓ Saved annotations: {annotation_file_path}")
        
        return {
            "detections": detections,
            "annotated_image_path": annotated_image_path,
            "annotation_file_path": annotation_file_path,
            "metadata_path": metadata_path,
            "num_detections": len(detections)
        }