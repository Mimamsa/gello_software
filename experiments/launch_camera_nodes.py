from dataclasses import dataclass
from multiprocessing import Process

import tyro
import time

#from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.cameras.usb_camera import USBCamera, get_sorted_v4l_paths, reset_all_elgato_devices

from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    hostname: str = "127.0.0.1"


def launch_server(port: int, camera_path: str, args: Args):
    camera = USBCamera(camera_path)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def main(args):
    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()

    # Wait for all v4l cameras to be back online
    time.sleep(0.1)
    
    v4l_paths = get_sorted_v4l_paths()
    #cameras = []
    camera_paths = []
    # store relevant paths
    # 1st slot for wrist camera, 2nd slot for base camera
    for p in v4l_paths:
        if 'Elgato' in p:
            camera_paths.append(p)
    #        cameras.append(USBCamera(p, 1280, 720, crop_frame=True))
    #for p in v4l_paths:
    #    if 'AVerMedia' in p:
    #        cameras.append(USBCamera(p, 640, 480))
    #print(cameras)

    camera_port = 5000
    camera_servers = []

    for camera_path in camera_paths:
        # start a python process for each camera
        print("Launching camera {} on port {}".format(camera_path, camera_port))
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_path, args))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
