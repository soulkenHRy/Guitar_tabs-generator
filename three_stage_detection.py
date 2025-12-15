"""
Three-Stage Hierarchical Guitar Detection System

Stage 1: Detect guitar (whole instrument) - GREEN
Stage 2: Detect neck within guitar region - BLUE  
Stage 3: Detect individual fret bars within neck region - YELLOW/RED

This hierarchical approach provides:
- Accurate guitar localization
- Precise neck region identification
- Fine-grained fret bar detection for finger tracking
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
from pathlib import Path
import torch


class ThreeStageGuitarDetector:
    """Neck + Fret Bars + Hand detector (no guitar stage)"""

    def __init__(self, neck_model_path, fret_bar_model_path,
                 device=None, imgsz=192, hold_frames=12, throttle_frets_every=1,
                 throttle_neck_every=1,
                 neck_segmentation_model_path=None, use_segmentation=False, seg_conf=0.3,
                 brightness_boost=1.0):
        print("Loading models...")

        if not Path(neck_model_path).exists():
            raise FileNotFoundError(f"Neck model not found: {neck_model_path}")
        if not Path(fret_bar_model_path).exists():
            raise FileNotFoundError(f"Fret bar model not found: {fret_bar_model_path}")
        self.neck_model = YOLO(neck_model_path)
        self.fret_bar_model = YOLO(fret_bar_model_path)
        
        print(f"✓ Neck Model: {neck_model_path}")
        print(f"✓ Fret Bars Model: {fret_bar_model_path}")
        print("All models loaded successfully!\n")

        # Runtime config (optimized for 15+ FPS)
        self.imgsz = imgsz
        self.device = device if device is not None else (0 if torch.cuda.is_available() else 'cpu')
        self.half = torch.cuda.is_available()
        self.max_hold_frames = hold_frames
        self.fret_throttle = throttle_frets_every
        self.neck_throttle = throttle_neck_every
        self.use_segmentation = use_segmentation
        self.seg_conf = seg_conf
        self.brightness_boost = brightness_boost

        # Persistence state
        self._last_neck_box = None
        self._last_neck_conf = 0.0
        self._neck_hold = 0
        self._last_fret_boxes = []
        self._fret_hold = 0
        self._frame_index = 0
        self._neck_frame = 0

        # Optional neck segmentation model (Ultralytics supports segmentation models too)
        self.neck_seg_model = None
        if self.use_segmentation and neck_segmentation_model_path:
            if not Path(neck_segmentation_model_path).exists():
                print(f"⚠ Neck segmentation model not found: {neck_segmentation_model_path}. Segmentation disabled.")
                self.use_segmentation = False
            else:
                print(f"✓ Neck Segmentation: {neck_segmentation_model_path}")
                self.neck_seg_model = YOLO(neck_segmentation_model_path)

    def preprocess_brightness(self, frame):
        """Apply brightness boost if enabled (helps in dim lighting)."""
        if self.brightness_boost == 1.0:
            return frame
        # Convert to float, boost, clip to valid range
        boosted = np.clip(frame.astype(np.float32) * self.brightness_boost, 0, 255).astype(np.uint8)
        return boosted

    def detect_stage2_neck(self, cropped_frame):
        results = self.neck_model.predict(cropped_frame, conf=0.25, verbose=False,
                                          imgsz=self.imgsz, device=self.device, half=self.half)
        return results[0].boxes

    def detect_stage3_fret_bars(self, cropped_neck):
        results = self.fret_bar_model.predict(cropped_neck, conf=0.2, verbose=False,
                                              imgsz=self.imgsz, device=self.device, half=self.half)
        return results[0].boxes

    def crop_region(self, frame, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(w, x2 + 5)
        y2 = min(h, y2 + 5)
        return frame[y1:y2, x1:x2], (x1, y1)

    def segment_neck_mask(self, neck_crop):
        """Return binary mask for neck crop if segmentation is enabled; else None."""
        if not self.use_segmentation or self.neck_seg_model is None:
            return None
        with torch.no_grad():
            seg_results = self.neck_seg_model.predict(neck_crop, conf=self.seg_conf, verbose=False,
                                                      imgsz=self.imgsz, device=self.device, half=self.half)
        if len(seg_results) == 0:
            return None
        r = seg_results[0]
        if not hasattr(r, 'masks') or r.masks is None:
            return None
        # Combine masks to a single binary mask
        mask = r.masks.data.sum(dim=0).cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def process_frame(self, frame):
        annotated_frame = frame.copy()
        stats = {
            'necks': 0,
            'fret_bars': 0,
            'details': [],
            'current_neck_detected': False,
            'frets_drawn': 0,
            'neck_conf': 0.0,
        }

        # Stage 2: single neck + persistence (throttled for FPS)
        run_neck = (self._neck_frame % self.neck_throttle == 0)
        self._neck_frame += 1
        
        # Preprocess frame for better detection in dim lighting
        processed_frame = self.preprocess_brightness(frame)
        
        if run_neck:
            with torch.no_grad():
                neck_boxes = self.detect_stage2_neck(processed_frame)
            stats['current_neck_detected'] = len(neck_boxes) > 0
            if len(neck_boxes) > 0:
                confs = neck_boxes.conf.cpu().numpy()
                idx = int(np.argmax(confs))
                self._last_neck_box = neck_boxes[idx]
                self._last_neck_conf = float(confs[idx])
                self._neck_hold = 0
                stats['neck_conf'] = self._last_neck_conf
            elif self._last_neck_box is not None and self._neck_hold < self.max_hold_frames:
                self._neck_hold += 1
            else:
                self._last_neck_box = None
                self._neck_hold = 0
        else:
            # Use cached neck box
            stats['current_neck_detected'] = self._last_neck_box is not None
            stats['neck_conf'] = self._last_neck_conf
            if self._last_neck_box is not None:
                self._neck_hold = min(self._neck_hold + 1, self.max_hold_frames)

        gstats = {'necks': 1 if self._last_neck_box is not None else 0, 'neck_details': []}

        if self._last_neck_box is None:
            stats['details'].append(gstats)
            return annotated_frame, stats

        nbox = self._last_neck_box
        nx1, ny1, nx2, ny2 = map(int, nbox.xyxy[0])
        # No guitar crop offsets in guitarless mode
        cv2.rectangle(annotated_frame, (nx1, ny1), (nx2, ny2), (255, 100, 0), 2)
        label = f'Neck #0 (conf:{stats["neck_conf"]:.2f})' + (" (hold)" if self._neck_hold > 0 else "")
        cv2.putText(annotated_frame, label,
                    (nx1, ny1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 2)

        # Crop neck region from full frame
        neck_crop, neck_offset = self.crop_region(frame, nbox)

        # Optional segmentation mask for neck
        neck_mask = self.segment_neck_mask(neck_crop)
        # Important: keep original crop for detection; use overlay only for visualization
        neck_crop_for_detect = neck_crop
        if neck_mask is not None:
            # Visualize mask overlay (semi-transparent) on a copy
            colored = neck_crop.copy()
            overlay = neck_crop.copy()
            overlay[neck_mask > 0] = (overlay[neck_mask > 0] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)
            alpha = 0.4
            neck_crop_visual = cv2.addWeighted(overlay, alpha, colored, 1 - alpha, 0)
            # Replace region in annotated_frame for visualization only
            h_vis, w_vis = neck_crop_visual.shape[:2]
            annotated_frame[ny1:ny1+h_vis, nx1:nx1+w_vis] = neck_crop_visual

        # Stage 3: fret bars (throttled + persistence)
        fret_boxes = []
        if self._frame_index % self.fret_throttle == 0:
            with torch.no_grad():
                fret_boxes = self.detect_stage3_fret_bars(neck_crop_for_detect)
            self._last_fret_boxes = fret_boxes
            self._fret_hold = 0
        elif len(self._last_fret_boxes) > 0 and self._fret_hold < self.max_hold_frames:
            fret_boxes = self._last_fret_boxes
            self._fret_hold += 1
        else:
            self._last_fret_boxes = []
            self._fret_hold = 0
        self._frame_index += 1

        fret_count = 0
        for fbox in fret_boxes:
            cls = int(fbox.cls[0])
            fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
            fx1 += neck_offset[0]
            fy1 += neck_offset[1]
            fx2 += neck_offset[0]
            fy2 += neck_offset[1]
            if cls == 0:
                # If we have a neck mask, tighten the fret box to mask region
                if neck_mask is not None:
                    # Map absolute coords back to neck_crop local coords
                    lx1 = max(0, int(fbox.xyxy[0][0].item()))
                    ly1 = max(0, int(fbox.xyxy[0][1].item()))
                    lx2 = min(neck_mask.shape[1]-1, int(fbox.xyxy[0][2].item()))
                    ly2 = min(neck_mask.shape[0]-1, int(fbox.xyxy[0][3].item()))
                    submask = neck_mask[ly1:ly2, lx1:lx2]
                    if submask.size > 0 and np.any(submask):
                        ys, xs = np.where(submask > 0)
                        if ys.size > 0:
                            tight_x1 = gx1 = guitar_offset[0] + neck_offset[0] + lx1 + int(xs.min())
                            tight_y1 = gy1 = guitar_offset[1] + neck_offset[1] + ly1 + int(ys.min())
                            tight_x2 = guitar_offset[0] + neck_offset[0] + lx1 + int(xs.max())
                            tight_y2 = guitar_offset[1] + neck_offset[1] + ly1 + int(ys.max())
                            fx1, fy1, fx2, fy2 = tight_x1, tight_y1, tight_x2, tight_y2
                fret_count += 1
                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 1)
            elif cls == 2:
                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, 'Nut' + (" (hold)" if self._fret_hold > 0 else ""),
                            (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        gstats['neck_details'].append({'neck_id': 0, 'fret_bars': fret_count})
        stats['fret_bars'] += fret_count
        stats['necks'] += 1 if self._last_neck_box is not None else 0
        stats['details'].append(gstats)
        stats['frets_drawn'] = fret_count

        return annotated_frame, stats


def main():
    # Model paths
    neck_model = "runs/detect/frets/weights/best.pt"
    fret_bar_model = "runs/detect/fret_bars/weights/best.pt"

    print("="*70)
    print("NECK + FRET DETECTION (Every Frame, 20 FPS Lock)")
    print("="*70)
    print("Stage: Detect Neck (BLUE boxes) - every frame")
    print("Stage: Detect Fret Bars (YELLOW) + Nut (RED) - every frame")
    print("="*70)
    print()

    for model_path, stage in [(neck_model, "Neck"), (fret_bar_model, "Fret Bars")]:
        if not Path(model_path).exists():
            print(f"⚠ {stage} model not found: {model_path}")
            if stage == "Neck":
                print("   python3 train_fret_simple.py")
                return
            if stage == "Fret Bars":
                print("   python3 train_fret_bars.py")
                return

    try:
        detector = ThreeStageGuitarDetector(
            neck_model,
            fret_bar_model,
            device=None,
            imgsz=192,
            hold_frames=12,
            throttle_frets_every=1,  # Run fret detection every frame
            throttle_neck_every=1,  # Run neck detection every frame
            neck_segmentation_model_path=None,
            use_segmentation=False,
            seg_conf=0.3,
            brightness_boost=1.5,  # Boost brightness 1.5x for dim lighting (1.0=off, 1.5-2.0 for low light)
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("\n✓ Camera opened successfully")
    print("\nPress 'q' to quit")
    print("="*70)
    print()

    # Frame rate limiter - lock to 20 FPS
    TARGET_FPS = 20
    FRAME_TIME = 1.0 / TARGET_FPS
    
    fps_time = time.time()
    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        annotated_frame, stats = detector.process_frame(frame)

        current_time = time.time()
        fps = 1.0 / (current_time - fps_time)
        fps_time = current_time
        
        # Limit frame rate to TARGET_FPS
        elapsed = time.time() - frame_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

        y_offset = 30
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(annotated_frame, f"Necks: {stats['necks']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        y_offset += 25
        cv2.putText(annotated_frame, f"Fret Bars: {stats['fret_bars']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        # Diagnostics overlay
        cv2.putText(annotated_frame, f"Curr Neck Detected: {'Yes' if stats.get('current_neck_detected') else 'No'}",
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(annotated_frame, f"Neck Conf: {stats.get('neck_conf', 0.0):.3f}",
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for guitar_stat in stats['details']:
            for neck_detail in guitar_stat['neck_details']:
                y_offset += 25
                text = f"  N{neck_detail['neck_id']}: {neck_detail['fret_bars']} frets"
                cv2.putText(annotated_frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Three-Stage Guitar Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection stopped")


if __name__ == "__main__":
    main()
