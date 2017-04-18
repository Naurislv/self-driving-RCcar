import RPi.GPIO as GPIO  # Import GPIO library
import time  # Import time library
import threading
import logging

GPIO.setmode(GPIO.BCM)  # Set GPIO pin numbering

TRIG = 24  # Associate pin 23 to TRIG
ECHO = 23  # Associate pin 24 to ECHO

logging.info("Distance measurement in progress")

GPIO.setup(TRIG, GPIO.OUT)  # Set pin as GPIO out
GPIO.setup(ECHO, GPIO.IN)  # Set pin as GPIO in

GPIO.output(TRIG, False)  # Set TRIG as LOW
logging.info("Waitng For 2 seconds for Sensor To Settle")
time.sleep(2)  # Delay of 2 seconds

meas = []  # Global list of ultrasonic sensor measurements


def add_measurement(m, size=10):
    """
    m: measurement
    """

    meas.append(m)
    if len(meas) > size:
        meas.pop(0)


def sensor_loop():
    oor = False

    while True:
        GPIO.output(TRIG, True)  # Set TRIG as HIGH
        time.sleep(0.0001)  # Delay of 0.00001 seconds
        GPIO.output(TRIG, False)  # Set TRIG as LOW

        while GPIO.input(ECHO) == 0:  # Check whether the ECHO is LOW
            pulse_start = time.time()  # Saves the last known time of LOW pulse

        pulse_duration = 0
        while GPIO.input(ECHO) == 1 and pulse_duration < 0.005:  # Check whether the ECHO is HIGH
            pulse_duration = time.time() - pulse_start  # Get pulse duration to a variable

        distance = pulse_duration * 17150  # Multiply pulse duration by 17150 to get distance
        distance = round(distance, 3)  # Round to two decimal points

        if distance < 40:  # Check whether the distance is within range
            if oor:
                add_measurement(999)
                oor = False
            else:
                add_measurement(distance)
        else:
            add_measurement(999)
            oor = True

        time.sleep(0.01)


class sensorThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        logging.info("Starting: " + self.name)
        sensor_loop()
        logging.info("Stopping: " + self.name)


thread1 = sensorThread(1, "Sensor Measurement Loop")
thread1.start()
