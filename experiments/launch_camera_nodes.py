from dataclasses import dataclass
from multiprocessing import Process

import tyro
import time

#from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.cameras.gopro_camera import GoProCamera, get_sorted_v4l_paths, reset_all_elgato_devices
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    hostname: str = "127.0.0.1"


def launch_server(port: int, camera_path: str, args: Args):
    camera = GoProCamera(camera_path)
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
    # pop non-relevant paths
    for i, p in enumerate(v4l_paths):
        if 'Elgato' not in p:
            print(v4l_paths.pop(i))

    camera_port = 5000
    camera_servers = []

    for camera_path in v4l_paths:
        # start a python process for each camera
        print(f"Launching camera {camera_path} on port {camera_port}")
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_path, args))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
