import os
import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import argparse

class SAHIInference:
    def __init__(self):
        self.detection_model = None
        self.font_path = 'Figtree-Regular.ttf'  # Optional: can use cv2.putText for faster rendering

    def draw_text(self, image, text, position, font_scale=0.6, color=(0,0,0), bg_color=(248,239,18,150), padding=4):
        """Draws text with background, uses cv2 as fast fallback"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            if self.font_path and os.path.exists(self.font_path):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image, "RGBA")
                font_size = int(font_scale * 20)
                font = ImageFont.truetype(self.font_path, font_size)
                x, y_bbox_top = position

                bbox = draw.textbbox((0,0), text, font=font)
                text_width = bbox[2]-bbox[0]
                text_height = bbox[3]-bbox[1]
                box_height = text_height + 2*padding
                box_width = text_width + 2*padding

                rect_bottom = y_bbox_top
                rect_top = rect_bottom - box_height
                rect_left = x - padding // 2
                rect_right = rect_left + box_width

                draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=bg_color)
                text_y = rect_top + padding // 2
                draw.text((x + padding, text_y), text, font=font, fill=color)

                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        # Fast fallback
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return image

    def load_model(self):
        weights = "rtdetr-x.pt"
        if not os.path.exists(weights):
            model = YOLO(weights)
            os.makedirs("models", exist_ok=True)
            if hasattr(model, 'save'):
                model.save(weights)

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights,
            device="cuda",
            confidence_threshold=0.25,
            image_size=640
        )

    def inference(self, source: str):
        exclude_classes = ['airplane', 'aeroplane', 'aircraft', 'plane', 'helicopter', 'drone', 'uav', 'jet']
        slice_width = slice_height = 640
        overlap_height_ratio = overlap_width_ratio = 0.1

        if not os.path.exists(source):
            print(f"Error: file '{source}' not found!")
            return

        cap = cv2.VideoCapture(source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.load_model()
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = get_sliced_prediction(
                rgb_frame,
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5,
                postprocess_class_agnostic=True
            )

            # Filter excluded classes
            filtered = [p for p in results.object_prediction_list
                        if not any(ex in p.category.name.lower() for ex in exclude_classes)]
            results.object_prediction_list = filtered

            annotated = frame.copy()
            count = len(results.object_prediction_list)

            for pred in results.object_prediction_list:
                bbox = pred.bbox
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                conf = pred.score.value
                cls_name = pred.category.name

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (18, 239, 248), 2)
                label = f"{cls_name}: {conf:.2f}"
                annotated = self.draw_text(annotated, label, (x1, y1))

            # Draw FPS top-left
            annotated = self.draw_text(annotated, f"FPS: {fps}", (10, 30), font_scale=1, bg_color=(255,255,255))
            
            # Draw object count top-right
            text = f"COUNT: {count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            annotated = self.draw_text(annotated, text, (width - text_size[0] - 10, 30), font_scale=1, bg_color=(255,255,255))

            # Show frame
            cv2.imshow("SAHI YOLO RTDETE-X Inference", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Inference completed. {frame_count} frames processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to video file")
    args = parser.parse_args()

    SAHIInference().inference(args.source)
