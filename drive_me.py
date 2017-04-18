"""Driving client.

To run this program in background :

sudo crontab -e
@reboot bash /home/pi/self-driving-RCcar/launcher.sh >/home/pi/self-driving-RCcar/logs/cronlog 2>&1

"""

import io
import os
from picamera import PiCamera
import pickle
import socket
import struct
import sys
import threading
import time
import logging

import controller
from ultrasonic_sensor_HCSR04 import SonicSensor

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

fps = sys.argv[1]
width = sys.argv[2]
height = sys.argv[3]


def uDistance():
    """Return ultrasonic sensor HCSR04 measurement data converted to centimeters."""
    return SonicSensor.measurements[-1]


class drive_me(object):
    """Main class."""

    def __init__(self):
        """Inicilize class variables."""
        # Connect a client socket to my_server:8000
        self.my_servers = ['192.168.10.254']  # '192.168.1.230'
        self.resolution = (int(width), int(height))  # 640,480 ; 320,200 ; 200, 66
        self.framerate = int(fps)
        self.rotation = 0

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

                    logging.debug('Sending image stream to server..')
                    with PiCamera() as camera:
                        camera.resolution = 640, 480
                        camera.framerate = self.framerate
                        camera.rotation = self.rotation
                        camera.zoom = (0.0, 0.3, 1.0, 0.455)  # x, y, w, h

                        # Let the camera warm up for 2 seconds
                        time.sleep(2)

                        # Note the start time and construct a stream to hold image data
                        # temporarily (we could write it directly to connection but in this
                        # case we want to find out the size of each capture first to keep
                        # our protocol simple)
                        stream = io.BytesIO()
                        counter = 0

                        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True,
                                                             resize=self.resolution):
                            # Write the length of the capture to the stream and flush to
                            # ensure it actually gets sent
                            stream.seek(0)

                            cp = self.server_time[counter]['client_process']
                            ans = self.server_time[counter]['server_time'] - time.time() - cp

                            data_string = pickle.dumps({'image': stream.read(),
                                                        'client_time': ans,
                                                        'uDistance': uDistance()})

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
                logging.exception()
                time.sleep(10)

    def recevei_thread(self, client_socket, counter):
        """Receive asnwer from server."""
        client_socket.settimeout(1.5)
        try:
            data = pickle.loads(client_socket.recv(1024))  # receive instructions from server

            steering, throttle = data['instruction']
            controller.steer_goal_set(steering)
            controller.speed_goal_set(throttle)

            self.server_time[counter + 1] = {}  # starts from 1
            self.server_time[counter + 1]['client_process'] = time.time()
            self.server_time[counter + 1]['server_time'] = data['server_time']

        except (EOFError, ConnectionResetError, socket.timeout):  # mostly catch socket.timeout when too FPS
            pass


if __name__ == "__main__":
    brain = drive_me()
    brain.drive()
