"""Driving client.

To run this program in background :

sudo crontab -e
@reboot bash /home/pi/self-driving-RCcar/launcher.sh >/home/pi/self-driving-RCcar/logs/cronlog 2>&1

"""
import argparse
import logging
import io
import os
from picamera import PiCamera
import pickle
import socket
import struct
import sys
import threading
import time
import psutil

# from ultrasonic_sensor_HCSR04 import SonicSensor


def uDistance():
    """Return ultrasonic sensor HCSR04 measurement data converted to centimeters."""
    # return SonicSensor.meas[-1]
    return 0


class drive_me(object):
    """Main class."""

    def __init__(self):
        """Inicilize class variables."""
        # Connect a client socket to my_server:8000
        self.my_servers = ['192.168.65.251']  # '192.168.1.230'
        self.resolution = (ARGS.width, ARGS.height)  # 640,480 ; 320,200 ; 200, 66
        self.framerate = ARGS.fps
        self.rotation = 180

    def server_address(self):
        """Ping addresses, if reachable then return it."""
        ret = []
        for addr in self.my_servers:
            response = os.system("ping -c 1 " + addr)
            if response == 0:
                ret.append(addr)
        return ret

    def drive(self):
        """Drive. Main Loop."""
        while True:
            try:
                self.server_time = {}
                self.server_time[0] = {}
                self.server_time[0]['client_process'] = 0
                self.server_time[0]['server_time'] = 0
                my_servers = self.server_address()

                for my_server in my_servers:
                    client_socket = socket.socket()
                    client_socket.connect((my_server, 8000))

                    # Make a file-like object out of the connection
                    connection = client_socket.makefile('wb')

                    logging.info('Sending image stream to server and starting motor driver..')
                    controller.START()
                    with PiCamera() as camera:
                        # resize image to closest resolution and then to get this resolution
                        camera.resolution = self.resolution
                        camera.framerate = self.framerate
                        camera.rotation = self.rotation
                        # camera.zoom = (0.0, 0.3, 1.0, 0.455)  # x, y, w, h

                        # Let the camera warm up for 2 seconds
                        time.sleep(2)

                        # Note the start time and construct a stream to hold image data
                        # temporarily (we could write it directly to connection but in this
                        # case we want to find out the size of each capture first to keep
                        # our protocol simple)
                        stream = io.BytesIO()
                        counter = 0

                        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                            # Write the length of the capture to the stream and flush to
                            # ensure it actually gets sent
                            stream.seek(0)

                            cp = self.server_time[counter]['client_process']
                            ans = self.server_time[counter]['server_time'] - time.time() - cp

                            # System Load. CPU load per core and memory usage.
                            cpu_load = psutil.cpu_percent(percpu=True)
                            mem_load = psutil.virtual_memory().percent
                            sys_load = [mem_load] + cpu_load

                            data_string = pickle.dumps({'image': stream.read(),
                                                        'client_time': ans,
                                                        'uDistance': uDistance(),
                                                        'sys_load': sys_load})

                            connection.write(struct.pack('<L', len(data_string)))
                            connection.flush()
                            # Rewind the stream and send the image data over the wire
                            connection.write(data_string)
                            threading.Thread(target=self.recevei_thread, args=[client_socket, counter]).start()

                            stream.seek(0)
                            stream.truncate()

                        # Write a length of zero to the stream to signal we're done
                        connection.write(struct.pack('<L', 0))
            except Exception as e:
                logging.exception(e)
                # After crash always se steer and speed to 0 so no phycal crash could occur
                logging.info('Halt!, Setting steer and speed to 0.')
                controller.HALT()
                time.sleep(10)
            except KeyboardInterrupt:
                logging.info('Setting steer and speed to 0.')
                controller.HALT()
                break

    def recevei_thread(self, client_socket, counter):
        """Receive asnwer from server."""
        client_socket.settimeout(1.5)
        try:
            data = pickle.loads(client_socket.recv(1024))  # receive instructions from server

            steering, throttle = data['instruction']
            controller.STEER_GOAL_SET(steering)
            controller.SPEED_GOAL_SET(throttle)

            self.server_time[counter + 1] = {}  # starts from 1
            self.server_time[counter + 1]['client_process'] = time.time()
            self.server_time[counter + 1]['server_time'] = data['server_time']

        except (EOFError, ConnectionResetError, socket.timeout):  # mostly catch socket.timeout when too FPS
            pass


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Actual autonomous driver placed on car itself.')
    PARSER.add_argument('--fps', type=int, default=7,
                        help='Frames per second for RasPI camera.')
    PARSER.add_argument('--width', type=int, default=280,
                        help='Width of RasPI camera video image.')
    PARSER.add_argument('--height', type=int, default=110,
                        help='Height of RasPI camera video image.')
    PARSER.add_argument('--controller', type=str, default='roomba',
                        help="Choose which controller you want to use. There are different"
                             "controllers for differnt hardware platforms.\n"
                             "Currently available : roomba or wltoys_a969")

    ARGS = PARSER.parse_args()

    # There may be different controllers for different robots, but for all of them,
    # they must support 4 commands:
    # controller.START()  # initialize system
    # controller.HALT()  # safetly shutdown system
    # controller.STEER_GOAL_SET()  # set steering goal as radius in mm
    # controller.SPEED_GOAL_SET()  # set speed goal cm/s

    if ARGS.controller == 'wltoys_a969':
        from wltoys_a969 import controller
    elif ARGS.controller == 'roomba':
        from roomba import controller

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    brain = drive_me()
    brain.drive()
