from __future__ import annotations

import cv2
import numpy as np


class BackgroundSegmenter:
    def __init__(self, history: int = 500, var_threshold: int = 50) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return mask
