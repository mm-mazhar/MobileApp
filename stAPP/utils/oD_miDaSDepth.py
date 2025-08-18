import cv2
import numpy as np
import torch

# models_dir = "./models"
# saved_model = os.path.join(models_dir, "efficientdet_lite0.tflite")


class MiDaSDepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.load_model()

    def load_model(self):
        self.model = (
            torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            .to(self.device)
            .eval()
        )
        transforms_module = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self.transform = (
            transforms_module.dpt_transform
            if "DPT" in self.model_type
            else transforms_module.small_transform
        )

    def estimate_depth(self, image_input):
        image = self.load_image(image_input)
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()
        return image, depth_map, self.create_colored_depth_map(depth_map)

    def load_image(self, image_input):
        img = cv2.imread(image_input)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def create_colored_depth_map(self, depth_map):
        d_min, d_max = depth_map.min(), depth_map.max()
        norm = (depth_map - d_min) / (d_max - d_min) if d_max > d_min else depth_map
        return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)


class CombinedObjectMiDaSDepth:
    def __init__(self, detection_model_path):
        self.depth_estimator = MiDaSDepthEstimator("MiDaS_small")
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=detection_model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options, score_threshold=0.5
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            from mediapipe.tasks.python.vision import Image as MPImage

            self.MPImage = MPImage
            self.ImageFormat = vision.ImageFormat
        except Exception as e:
            print(f"Object detection not available: {e}")
            self.detector = None

    def detect_objects(self, image_input, depth_map):
        original = self.depth_estimator.load_image(image_input)
        if not self.detector:
            print("⚠️ Detector not initialized")
            return cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        mp_image = self.MPImage(image_format=self.ImageFormat.SRGB, data=original)
        detection_result = self.detector.detect(mp_image)

        annotated = original.copy()
        if not detection_result.detections:
            print("⚠️ No objects detected.")
            return cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            cat = detection.categories[0].category_name
            x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
            x2, y2 = int(x1 + bbox.width), int(y1 + bbox.height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(region)
            color = self.depth_to_color(avg_depth, depth_map.min(), depth_map.max())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{cat}: {avg_depth:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    def depth_to_color(self, value, min_d, max_d):
        norm = (value - min_d) / (max_d - min_d) if max_d > min_d else 0.5
        return (int(255 * (1 - norm)), 50, int(255 * norm))  # Red-to-blue
