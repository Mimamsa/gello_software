from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import cv2


def viz_frame(
    obs: Dict,
    camera_names: List,
    viz_res: Tuple = None,
    text: str = None
) -> None:
    """Visualize single frame
    Args
        obs (Dict): Environment observations.
        camera_names (List[str]): Camera names.
        viz_res (Tuple[int, int]): Target image resolution to be visualized (w,h)
    """
    for name in camera_names:
        key = f"{name}_rgb"
        cv2.namedWindow(key, cv2.WINDOW_AUTOSIZE)  
        vis_img = obs[key].copy()
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # resize frame (960, 720)->(?)
        if viz_res:
            cv2.resize(vis_img, viz_res)

        if text:
            cv2.putText(vis_img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow(key, vis_img)
        cv2.moveWindow(key, 0, 0)
        cv2.pollKey()
