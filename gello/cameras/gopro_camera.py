import os
import time
from typing import List, Optional, Tuple

import numpy as np
import cv2
import pathlib
from subprocess import Popen, PIPE, DEVNULL
import fcntl

from gello.cameras.camera import CameraDriver


def create_usb_list():
    device_list = list()
    lsusb_out = Popen('lsusb -v', shell=True, bufsize=64, 
                      stdin=PIPE, stdout=PIPE, stderr=DEVNULL, 
                      close_fds=True).stdout.read().strip().decode('utf-8')
    usb_devices = lsusb_out.split('%s%s' % (os.linesep, os.linesep))
    for device_categories in usb_devices:
        if not device_categories:
            continue
        categories = device_categories.split(os.linesep)
        device_stuff = categories[0].strip().split()
        bus = device_stuff[1]
        device = device_stuff[3][:-1]
        device_dict = {'bus': bus, 'device': device}
        device_info = ' '.join(device_stuff[6:])
        device_dict['description'] = device_info
        for category in categories:
            if not category:
                continue
            categoryinfo = category.strip().split()
            if categoryinfo[0] == 'iManufacturer':
                manufacturer_info = ' '.join(categoryinfo[2:])
                device_dict['manufacturer'] = manufacturer_info
            if categoryinfo[0] == 'iProduct':
                device_info = ' '.join(categoryinfo[2:])
                device_dict['device'] = device_info
        path = '/dev/bus/usb/%s/%s' % (bus, device)
        device_dict['path'] = path

        device_list.append(device_dict)
    return device_list

def reset_usb_device(dev_path):
    USBDEVFS_RESET = 21780
    try:
        f = open(dev_path, 'w', os.O_WRONLY)
        fcntl.ioctl(f, USBDEVFS_RESET, 0)
        print('Successfully reset %s' % dev_path)
    except PermissionError as ex:
        raise PermissionError('Try running "sudo chmod 777 {}"'.format(dev_path))

def reset_all_elgato_devices():
    """
    Find and reset all Elgato capture cards.
    Required to workaround a firmware bug.
    """
    
    # enumerate UBS device to find Elgato Capture Card
    device_list = create_usb_list()
    
    for dev in device_list:
        if 'Elgato' in dev['description']:
            dev_usb_path = dev['path']
            reset_usb_device(dev_usb_path)


def get_sorted_v4l_paths(by_id=True):
    """
    If by_id, sort devices by device name + serial number (preserves device order)
    else, sort devices by usb bus id (preserves usb port order)
    Args
        by_id (bool):
    Returns
        (list[str]):
    """
    
    dirname = 'by-id'
    if not by_id:
        dirname = 'by-path'
    v4l_dir = pathlib.Path('/dev/v4l').joinpath(dirname)

    valid_paths = list()
    for dev_path in sorted(v4l_dir.glob("*video*")):
        name = dev_path.name

        # only keep devices ends with "index0"
        # since they are the only valid video devices
        index_str = name.split('-')[-1]
        assert index_str.startswith('index')
        index = int(index_str[5:])
        if index == 0:
            valid_paths.append(dev_path)

    result = [str(x.absolute()) for x in valid_paths]

    return result


class GoProCamera(CameraDriver):
    """
    https://help.elgato.com/hc/en-us/articles/360027952992-Supported-resolutions-for-Elgato-Game-Capture-HD
    """

    def __repr__(self) -> str:
        return f"GoProCamera(device_path={self.device_path})"

    def __init__(
        self,
        device_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        cap_buffer_size: int = 1,
        fps: int = 60
    ):

        self.device_path = device_path
        self.width = width
        self.height = height
        self.cap_buffer_size = cap_buffer_size
        self.fps = fps

        if device_path is None:
            # Find and reset all Elgato capture cards.
            # Required to workaround a firmware bug.
            reset_all_elgato_devices()

            # Wait for all v4l cameras to be back online
            time.sleep(0.1)
            v4l_paths = get_sorted_v4l_paths()

            # pop non-relevant paths
            for i, p in enumerate(v4l_paths):
                if 'Elgato' not in p:
                    print(v4l_paths.pop(i))
            print('v4l_paths: ', v4l_paths)

            # Assume only one GoPro camera exists
            self.device_path = v4l_paths[0]

        self._cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)  # 1920
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)  # 1080
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Find frame crop parameters
        self._resized_w = int(self.height/3*4)
        self._w_offset = int((self.width-self._resized_w)/2)


    def __del__(self):
        self._cap.release()

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return (w,h). If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        if self._cap.isOpened():
            ret = self._cap.grab()
            ret, frame = self._cap.retrieve()
        else:
            print('\'self._cap\' is not opened.')
            return
        
        if img_size is None:
            image = frame[:,:,::-1]  # bgr -> rgb
        else:
            image = cv2.resize(frame, img_size)[:,:,::-1]  # bgr -> rgb
            
        # Crop image resolution (w,h) from (1280,720) to (960,720)
        image = image[:, self._w_offset:self._w_offset+self._resized_w, :]

        return image, image[:, :, 0]
