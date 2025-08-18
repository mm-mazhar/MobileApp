import torch
import cv2
import numpy as np


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
