"""Server."""
import argparse
import cv2
from datetime import datetime
from keras.models import model_from_json
import numpy as np
import os
import pickle
# from scipy.misc import imresize
import socket
import struct
import tensorflow as tf
import threading
import time
import usb.core
import usb.util
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description='Autonomous Driving Server')
parser.add_argument('--autonomous', type=bool, default=False,
                    help='Boolean. If True then will be using Autonomous car capabilities.')

args = parser.parse_args()


class ExitTkinterWindow(object):
    """Custom exit error."""

    pass


class server(object):
    """Network server."""

    def __init__(self):
        """Main server loop. Create socket and waitinf for connections."""
        # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
        # all interfaces)

        with open('calib.p', 'rb') as f:
            self.calib_params = pickle.load(f)   # calibration parameters for undistortion
        logging.info('Calibration parameters loaded.')

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 8000))
        server_socket.listen(0)

        self.drivers = {}

        threading.Thread(target=self.instructor_G27).start()
        while True:
            # accept connections from outside
            logging.info('Waiting for new connection..')
            (clientsocket, address) = server_socket.accept()
            # now do something with the clientsocket
            # in this case, we'll pretend this is a threaded server
            logging.info('Connection from {}'.format(address))
            threading.Thread(target=self.read_images, args=(clientsocket, address)).start()
            time.sleep(2)

    def instructor_G27(self):
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
            logging.info('Steering wheel not found!')
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
                            addresses = sorted(list(self.drivers.keys()))  # get all 1 level keys (addresses)
                            try:
                                address = addresses[idx]
                                constant_speed = 0  # to zero when selecting new driver
                                logging.info('Selecting {} to drive'.format(address))

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
                                os.makedirs(working_dir + dir_name)
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

                    if address != '' and mode == 'train':
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
                            self.drivers[address]['commands'] = (False, position, 255 - throttle, 255 - brakes)
                        else:
                            self.drivers[address]['commands'] = (True, position, constant_speed, 255 - brakes)

                except usb.core.USBError as e:
                    data = None
                    if e.args == ('Operation timed out',):
                        continue

    def read_images(self, clientsocket, address):
        """Read images in loop directly from connection."""
        self.drivers[address[0]] = {}
        self.drivers[address[0]]['commands'] = (False, 0, 0, 0)
        self.drivers[address[0]]['save'] = False
        self.drivers[address[0]]['mode'] = 'testing'

        clientsocket.settimeout(10)
        connection = clientsocket.makefile('rb')

        counter = 0
        times = []  # to calculate FPS depending on history data
        th = None
        logging.info('Entering global loop')

        while True:
            try:
                start_time = time.time()
                times.append(start_time)
                if len(times) > 10:
                    times.pop(0)

                # Read the length of the image as a 32-bit unsigned int.
                image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                # Construct a stream to hold the image data and read the image data from the connection
                data = pickle.loads(connection.read(image_len))

                network_latency = time.time() + data['client_time']
                fps = round(len(times) / (time.time() - times[0]), 0)

                if th is not None:
                    th.join()  # Keras wont work with more than one thread
                th = threading.Thread(target=self.process_data, args=[clientsocket, address, data,
                                                                      network_latency, fps, counter])
                th.start()

            except Exception as e:
                # logging.exception(e)
                logging.info('Display window closed. Closing connection. {}'.format(address[0]))
                break

            counter += 1

        del self.drivers[address[0]]
        connection.close()
        clientsocket.close()
        logging.info('connection closed {}'.format(address[0]))

    def steering_angle(self, steering, max_steering, adjust_output=None):
        """Convert steering values to normalized steering angle."""
        if steering < -max_steering:
            steering = -max_steering
        elif steering > max_steering:
            steering = max_steering

        output = steering / (max_steering + 1)
        if adjust_output is None:
            return output
        else:
            return output * adjust_output

    def driving_speed(self, address, throttle, max_speed):
        """Convert throttle values to normalized driving speed."""

        self.drivers[address]['speed'] += 1 * throttle

        if self.drivers[address]['speed'] > max_speed:
            return max_speed

        return self.drivers[address]['speed']

    def check_safety(self, steering, throttle, uDistance):

        # if uDistance < 40:
        #     return 0, 1

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
        uDistance = data['uDistance']  # Ultrasonic sensor measurements in cm
        sys_load = data['sys_load']  # Client CPU load

        image = cv2.undistort(image, self.calib_params['mtx'], self.calib_params['dist'],
                              None, self.calib_params['mtx'])

        max_steering = 20  # max value passed to servo motor
        max_speed = 30  # max value passed to servo motor
        min_speed = 3  # min value passed to servo motor

        if self.drivers[address[0]]['mode'] == 'autonomous' and args.autonomous:
            # cv2 load image as BGR, but model expects RGB and crop center image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, 40:240]

            with graph.as_default():
                steering = float(model.predict(image[None, :, :, :], batch_size=1)[0, 0] * max_steering + 10)
                throttle = float(self.driving_speed(address[0], 0.08, max_speed))
        else:
            const_speed, steering, throttle, brakes = self.drivers[address[0]]['commands']

            if brakes == 0 and not const_speed:
                throttle_noise = 52  # expect throttle to be from 0 .. 255
                if throttle <= 52:
                    # to remove fluctuactions when clutch is not completely relaxed
                    throttle = 0
                else:
                    # now throttle will be from min_speed ... max_speed
                    throttle = (throttle - throttle_noise) / (255 - throttle_noise)
                    throttle = throttle * (max_speed - min_speed) + min_speed
            elif brakes == 0 and const_speed:
                pass  # use constant speed as is
            else:
                throttle = -brakes / 255 * max_speed

            steering = self.steering_angle(steering, 10000, max_steering)

        steering, throttle = self.check_safety(steering, throttle, uDistance)  # Safety system. Avoid collisions etc.
        clientsocket.sendall(pickle.dumps({'counter': counter, 'server_time': time.time(),
                                           'instruction': (steering, throttle)}))

        processing_time = time.time() - start_time
        threading.Thread(target=self.display_image, args=(image, address[0], processing_time, network_latency,
                                                          steering, throttle, fps, counter, uDistance,
                                                          sys_load)).start()

    def display_image(self, image, address, processing_time, network_latency,
                      steering, throttle, fps, counter, uDistance, sys_load):
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
        y0, dy = 15, 15
        for i, t in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(im_resized, t, (10, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(im_resized, 'angle: ' + str(steering), (10, 135), font, 0.7, (0, 153, 76), 1, cv2.LINE_AA)
        cv2.putText(im_resized, 'speed: ' + str(throttle), (10, 165), font, 0.7, (0, 128, 255), 1, cv2.LINE_AA)
        cv2.putText(im_resized, 'collision: ' + str(uDistance), (10, 195), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)

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
        cv2.imwrite(working_dir + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
                                                                   'L', instruction), image[:, 0:200])
        # im_center
        cv2.imwrite(working_dir + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
                                                                   'C', instruction), image[:, 40:240])
        # im_right
        cv2.imwrite(working_dir + to_dir + '/{}_{}_{}.jpeg'.format(add + str(counter),
                                                                   'R', instruction), image[:, 80:280])


if __name__ == "__main__":
    working_dir = '/home/nauris/Downloads/images/'
    model_path = 'keras_models/model.json'

    if args.autonomous:
        with open(model_path, 'r') as jfile:
            model = model_from_json(jfile.read())
            logging.info('Model loaded')

        model.compile("adam", "mse")
        graph = tf.get_default_graph()
        logging.info('Model compiled')
        weights_file = model_path.replace('json', 'h5')
        model.load_weights(weights_file)
        logging.info('Weights loaded')
    else:
        logging.info('No autonomous driving features.')

    server()
