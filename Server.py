"""Server."""

# Standard imports
import argparse
import os
import pickle
from datetime import datetime
import socket
import struct
import threading
import time
import logging

# Local imports
from keras.models import model_from_json
from Keystroke import wait_key, STROKES

# Other Imports
import cv2
import numpy as np
import tensorflow as tf
import usb.core
import usb.util

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

PARSER = argparse.ArgumentParser(description='Autonomous Driving Server')
PARSER.add_argument('--autonomous', type=bool, default=False,
                    help='Boolean. If True then will be using Autonomous car capabilities.')

PARSER.add_argument('--undistort', type=bool, default=False,
                    help="Boolean. If True then will undistort raspberry image sent from "
                         "Fisheye lense.")

ARGS = PARSER.parse_args()


class Server(object):
    """Network server."""

    def __init__(self):
        """Main server loop. Create socket and waitinf for connections."""
        # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
        # all interfaces)

        if ARGS.undistort:
            with open('calib.p', 'rb') as calib_file:
                # calibration parameters for undistortion
                self.calib_params = pickle.load(calib_file)
            logging.info('Calibration parameters loaded.')

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 8000))
        server_socket.listen(0)

        self.drivers = {}

        threading.Thread(target=self.instructor_g27).start()
        while True:
            # accept connections from outside
            logging.info('Waiting for new connection..')
            (clientsocket, address) = server_socket.accept()
            # now do something with the clientsocket
            # in this case, we'll pretend this is a threaded server
            logging.info('Connection from %s', address)
            threading.Thread(target=self.read_images, args=(clientsocket, address)).start()
            time.sleep(2)

    def instructor_keyboard(self):
        """Give commands to driver from keyboard."""

        recv = True
        address = ''
        idx = 0

        while recv is not False:
            recv = wait_key()

            address_list = sorted(list(self.drivers.keys()))

            if address_list and recv is not None: # if address is not empty list

                if recv == 'next':
                    address = address_list[idx]
                    logging.info('Connected to %s driver', address)
                    idx += 1
                    if idx == len(address_list):
                        idx = 0
                elif recv == 'mode':
                    if self.drivers[address]['mode'] != 'train':
                        logging.info('Entering Train mode')
                        self.drivers[address]['mode'] = 'train'
                        self.drivers[address]['speed'] = 0
                    elif self.drivers[address]['mode'] == 'train':
                        logging.info('Entering Autonomous mode')
                        self.drivers[address]['mode'] = 'autonomous'
                        self.drivers[address]['speed'] = 0
                elif recv == 'left':
                    # Inverse steering radius (isr) 1/r used to make steering independet
                    # of car geometry. When raidus goes towards infinity (straigth)
                    # line then isr goes towards 0. Radius is measured in meters.
                    current_isr = self.drivers[address]['commands'][0]

                    # convert to radius change value in meters for convenience
                    # current_radius = 1 / current_isr
                    # set_radius = current_radius + 0.1

                    current_isr -= 0.1
                    self.drivers[address]['commands'][0] = current_isr

                    logging.debug('Radius in meters set to %s', current_isr)
                elif recv == 'right':
                    # Inverse steering radius (isr) 1/r used to make steering independet
                    # of car geometry. When raidus goes towards infinity (straigth)
                    # line then isr goes towards 0. Radius is measured in meters.
                    current_isr = self.drivers[address]['commands'][0]

                    # convert to radius change value in meters for convenience
                    # current_radius = 1 / current_isr
                    # set_radius = current_radius + 0.1

                    current_isr += 0.1
                    self.drivers[address]['commands'][0] = current_isr

                    logging.debug('Radius in meters set to %s', current_isr)
                elif recv == 'forward':
                    # Speed is measured in km/h
                    self.drivers[address]['commands'][1] += 0.2
                elif recv == 'backward':
                    # Speed is measured in km/h
                    self.drivers[address]['commands'][1] -= 0.2
                elif recv == 'stop':
                    # Speed is measured in km/h
                    self.drivers[address]['commands'][1] = 0

            elif recv is not None:
                logging.info("There is no driver connected right now.")
            else:
                logging.info(address)
                logging.info(self.drivers)
                logging.info("Command button does not exist see full list"
                             "of commands available")

                for key, val in STROKES.items():
                    logging.info("   %s: %s", repr(key), val)

    def instructor_g27(self):
        """Driving instructor with Logitech G27.

        controls_0 :
            0
            1 when pressed right shift
            2 when pressed left shift
            4 when pressed up right button
            8 when pressed up left button
        """
        # 0xc29b ; 0xc52b
        dev = usb.core.find(idVendor=0x046d, idProduct=0xc29b)

        if dev is None:
            logging.info('Steering wheel not found, fallback to keyboard.')
            self.instructor_keyboard()
        else:
            logging.info('Logitech G27 Racing Wheel found!')

            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
            endpoint = dev[0][(0, 0)][0]

            idx = 0
            address = ''
            clicked = 0
            constant_speed = 0

            while True:
                try:
                    start_time = time.time()
                    data = dev.read(endpoint.bEndpointAddress,
                                    endpoint.wMaxPacketSize)
                    controls_0 = data[1]
                    controls_1 = data[2]
                    steering_angle = data[3]
                    throttle = data[5]
                    brakes = data[6]
                    # clutch = data[7]

                    if controls_0 == 1:
                        diff = start_time - clicked
                        if diff > 0.3:
                            # get all 1 level keys (addresses)
                            addresses = sorted(list(self.drivers.keys()))
                            try:
                                address = addresses[idx]
                                constant_speed = 0  # to zero when selecting new driver
                                logging.info('Selecting %s to drive', address)

                                idx += 1
                            except IndexError:
                                address = ''
                                idx = 0
                        clicked = time.time()
                    elif controls_0 == 4 and address != '':
                        diff = start_time - clicked
                        if diff > 0.3:
                            if self.drivers[address]['save']:
                                logging.info('Saving files : Stoped')
                                self.drivers[address]['save'] = False
                            else:
                                logging.info('Saving files : Started')
                                dir_name = datetime.now().strftime("%Y-%m-%d-%I:%M:%S")
                                os.makedirs(WORKING_DIR + dir_name)
                                self.drivers[address]['save_dir'] = dir_name
                                self.drivers[address]['save'] = True
                        clicked = time.time()
                    elif controls_0 == 2 and address != '':
                        diff = start_time - clicked
                        if diff > 0.3:
                            if self.drivers[address]['mode'] != 'train':
                                logging.info('Entering Train mode')
                                self.drivers[address]['mode'] = 'train'
                                pos_0 = steering_angle
                                prev_position = steering_angle
                                adder = 0  # when go past 255 value and start over
                            elif self.drivers[address]['mode'] == 'train':
                                logging.info('Entering Autonomous mode')
                                self.drivers[address]['speed'] = 0
                                self.drivers[address]['mode'] = 'autonomous'
                        clicked = time.time()
                    elif controls_1 == 64 and address != '':
                        diff = start_time - clicked
                        if diff > 0.3:
                            constant_speed += 1
                        clicked = time.time()
                    elif controls_1 == 128 and address != '':
                        diff = start_time - clicked
                        if diff > 0.3:
                            constant_speed -= 1
                        clicked = time.time()

                    try:
                        mode = self.drivers[address]['mode']
                    except KeyError:
                        address = ''
                        idx = 0
                        continue

                    if address != '' and mode != 'testing':
                        diff_pos = steering_angle - prev_position
                        if abs(diff_pos) > 120 and diff_pos < 0:
                            adder += 255
                        elif abs(diff_pos) > 120 and diff_pos > 0:
                            adder += -255

                        position = steering_angle - pos_0 + adder
                        prev_position = steering_angle

                        if brakes < 255:
                            constant_speed = 0  # to reset constant_speed to 0

                        if constant_speed == 0:
                            cmd = (False, position, 255 - throttle, 255 - brakes)
                            self.drivers[address]['commands'] = cmd
                        else:
                            cmd = (True, position, constant_speed, 255 - brakes)
                            self.drivers[address]['commands'] = cmd
                    elif address == '':
                        mode = 'testing'

                except usb.core.USBError as err:
                    data = None
                    if err.args == ('Operation timed out',):
                        continue

    def read_images(self, clientsocket, address):
        """Read images in loop directly from connection."""
        self.drivers[address[0]] = {}
        self.drivers[address[0]]['commands'] = [0, 0]
        self.drivers[address[0]]['save'] = False
        self.drivers[address[0]]['mode'] = 'testing'

        clientsocket.settimeout(10)
        connection = clientsocket.makefile('rb')

        counter = 0
        times = []  # to calculate FPS depending on history data
        thrd = None
        logging.info('Entering global loop')

        while True:
            try:
                start_time = time.time()
                times.append(start_time)
                if len(times) > 10:
                    times.pop(0)

                # Read the length of the image as a 32-bit unsigned int.
                image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                # Construct a stream to hold the image data and read the image data
                # from the connection
                data = pickle.loads(connection.read(image_len))

                network_latency = time.time() + data['client_time']
                fps = round(len(times) / (time.time() - times[0]), 0)

                if thrd is not None:
                    thrd.join()  # Keras wont work with more than one thread
                thrd = threading.Thread(target=self.process_data,
                                        args=[clientsocket, address, data,
                                              network_latency, fps, counter])
                thrd.start()

            except Exception:
                # logging.exception("While loop exception")
                logging.info('Display window closed. Closing connection. %s', address[0])
                break

            counter += 1

        del self.drivers[address[0]]
        connection.close()
        clientsocket.close()
        logging.info('connection closed %s', address[0])

    def check_safety(self, steering, throttle, u_distance):
        """Check Safety with ultrasonic sensor."""

        if u_distance < 40:
            return 0, 1

        return steering, throttle

    def process_data(self, clientsocket, address, data, network_latency, fps, counter):
        """Image processing.

        1. Image loading
        2. Image processing
        3. Image sending back to car
        4. Image displaying (other thread)
        """

        start_time = time.time()
        image = cv2.imdecode(np.fromstring(data['image'], np.uint8), 1)  # image to be saved
        u_distance = data['uDistance']  # Ultrasonic sensor measurements in cm
        sys_load = data['sys_load']  # Client CPU load

        if ARGS.undistort:
            image = cv2.undistort(image, self.calib_params['mtx'],
                                  self.calib_params['dist'],
                                  None, self.calib_params['mtx'])

        steering, throttle = self.drivers[address[0]]['commands']

        if self.drivers[address[0]]['mode'] == 'autonomous' and ARGS.autonomous:
            # cv2 load image as BGR, but model expects RGB and crop center image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, 40:240]

            with GRAPH.as_default():
                pred = MODEL.predict(image[None, :, :, :], batch_size=1)[0, 0]
                # current trained model predicts number in interval 0 .. 1 which
                # is steering angle. We need to convert it to inverse radius. Which
                # of-course if rough estimation.

                # s = 0.01  # wheel base
                # a = pred * 18  # steering wheel angle
                # n = 1  # steering ratio (e.g. for 16:1, n = 16)

                # radius = s / (sqrt(2 - 2 * cos(2*a/n)))
                steering = float(pred)

        # logging.info("%s %s", steering, throttle)
        # Safety system. Avoid collisions etc.
        # steering, throttle = self.check_safety(steering, throttle, uDistance)
        clientsocket.sendall(pickle.dumps({'counter': counter, 'server_time': time.time(),
                                           'instruction': (steering, throttle)}))

        processing_time = time.time() - start_time
        threading.Thread(target=self.display_image,
                         args=(image, address[0], processing_time, network_latency,
                               steering, throttle, fps, counter, u_distance,
                               sys_load)).start()

    def display_image(self, image, address, processing_time, network_latency,
                      steering, throttle, fps, counter, u_distance, sys_load):
        """"Display image alongside with necessary information with cv2."""
        imshape = image.shape
        text = ("FPS: {}\nNetwork: {}ms"
                "\nImage processing: {}ms"
                "\nIP: {}\nShape: {}\n"
                "sys_load: {}").format(fps, round(network_latency, 3) * 1000,
                                       round(processing_time, 3) * 1000, address,
                                       imshape, sys_load)

        new_width = 1500
        new_hight = imshape[0] * new_width // imshape[1]
        im_resized = cv2.resize(image, (new_width, new_hight))

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_0, d_y = 15, 15
        for i, txt in enumerate(text.split('\n')):
            y_coord = y_0 + i * d_y
            cv2.putText(im_resized, txt, (10, y_coord), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(im_resized, 'angle: ' + str(steering), (10, 135), font, 0.7,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(im_resized, 'speed: ' + str(throttle), (10, 165), font, 0.7,
                    (0, 128, 255), 1, cv2.LINE_AA)
        cv2.putText(im_resized, 'collision: ' + str(u_distance), (10, 195), font, 0.7,
                    (255, 0, 0), 1, cv2.LINE_AA)

        cv2.startWindowThread()
        cv2.namedWindow(address)
        cv2.imshow(address, im_resized)

        if self.drivers[address]['save']:
            self.save_image(image, counter, address, str((steering, throttle)))

    def save_image(self, image, counter, address, instruction):
        """Save image."""
        to_dir = self.drivers[address]['save_dir']

        add = '0'
        if counter < 10:
            add = '00000'
        elif counter < 100:
            add = '0000'
        elif counter < 1000:
            add = '000'
        elif counter < 10000:
            add = '00'

        # cv2.imwrite(working_dir + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
        #                                                            'C', instruction), image)
        # im_left
        cv2.imwrite(WORKING_DIR + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
                                                                   'L', instruction),
                    image[:, 0:200])
        # im_center
        cv2.imwrite(WORKING_DIR + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
                                                                   'C', instruction),
                    image[:, 40:240])
        # im_right
        cv2.imwrite(WORKING_DIR + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
                                                                   'R', instruction),
                    image[:, 80:280])


if __name__ == "__main__":
    WORKING_DIR = '/home/nauris/Downloads/images/'
    MODEL_PATH = 'models/model.json'

    if ARGS.autonomous:
        with open(MODEL_PATH, 'r') as jfile:
            MODEL = model_from_json(jfile.read())
            logging.info('Model loaded')

        MODEL.compile("adam", "mse")
        GRAPH = tf.get_default_graph()
        logging.info('Model compiled')
        WEIGHTS_FILE = MODEL_PATH.replace('json', 'h5')
        MODEL.load_weights(WEIGHTS_FILE)
        logging.info('Weights loaded')
    else:
        logging.info('No autonomous driving features.')

    Server()
