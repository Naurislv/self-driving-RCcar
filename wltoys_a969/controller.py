"""Controls for Wltoys A969 model car motor/steering actuators (from Raspberry Pi).

Simple test program:
    import time
    import wltoys_a969

    time.sleep(1)

    wltoys_a969.steer_goal_set(10)  # Angle in [STEER_MIN_ANGLE..STEER_MAX_ANGLE] (in degrees)
    wltoys_a969.speed_goal_set(0)  # Speed in [-MOTOR_PWM_MAX..MOTOR_PWM_MAX] (slower..faster)
    time.sleep(1.3)

    wltoys_a969.steer_goal_set(10)
    wltoys_a969.speed_goal_set(20)
    time.sleep(1.3)

    wltoys_a969.steer_goal_set(0)
    wltoys_a969.speed_goal_set(0)
    time.sleep(5.0)


    wltoys_a969.halt()
"""

import RPi.GPIO as IO
import Adafruit_ADS1x15
import time
import threading
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


IO.setwarnings(False)
IO.setmode(IO.BCM)

IO.setup(19, IO.OUT)
IO.output(19, IO.LOW)
pin_motor_fwd = IO.PWM(19, 2250)
pin_motor_fwd.start(0)

IO.setup(16, IO.OUT)
IO.output(16, IO.LOW)
pin_motor_bwd = IO.PWM(16, 2250)
pin_motor_bwd.start(0)

IO.setup(26, IO.OUT)
IO.output(26, IO.LOW)
pin_steer_right = IO.PWM(26, 100)
pin_steer_right.start(0)

IO.setup(20, IO.OUT)
IO.output(20, IO.LOW)
pin_steer_left = IO.PWM(20, 100)
pin_steer_left.start(0)

adc = Adafruit_ADS1x15.ADS1015()

# parameters
ADC_GAIN = 1
MOTOR_PWM_MAX = 30  # Max speed sent to motor
STEER_MIN = 1123  # Steer servo min feedback value
STEER_MAX = 837  # Steer servo max feedback value
STEER_PWM_T = 25 / 2.55  # Steer servo threshold PWM abs value
STEER_PWM_MAX = (255 - STEER_PWM_T) / 2.55  # Steer servo max PWM abs value
STEER_MIN_ANGLE = -20.0  # Steer servo min angle value
STEER_IDLE_ANGLE = 0.0  # Steer servo idle angle value
STEER_MAX_ANGLE = 20.0  # Steer servo max angle value


def gaussian(size, sig, min_val=3, max_val=20):
    mu = 0
    x = np.linspace(mu - 0.15, 2, size)
    dist = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * max_val

    return dist[dist > min_val]


impulse_dist = gaussian(350, 0.3, min_val=4, max_val=30)
logging.info('Impulse Dist {}'.format(impulse_dist))


class DriverController(object):
    """Controller/Driver class which is specificly written to control specific hardware."""

    def __init__(self):
        # goal values to set externally
        self.goal_speed = 0
        self.goal_steer_angle = 0
        self.impulse = []

        self.running = False

    def act_loop(self):

        while True:
            try:
                if self.running:
                    self.steer_step(self.goal_steer_angle)
                    goal_speed = self.goal_speed

                    if len(self.impulse) > 0:
                        if goal_speed < self.impulse[0]:
                            goal_speed = self.impulse[0]  # add starting impulse to goal speed

                        # one by one remove impulse values until it's empty list
                        self.impulse = np.delete(self.impulse, 0)

                    self.motor_step(goal_speed)
                    time.sleep(0.0001)
                else:
                    time.sleep(1)
            except Exception as e:
                logging.exception(e)
                break

        logging.info('Stop Work. Set speed goal and steer goal to 0')
        self.speed_goal_set(0)
        self.steer_goal_set(0)

    def halt(self):
        self.running = False

    def start(self):
        self.running = True

    def speed_goal_set(self, speed):
        """Set New goal for speed and check if starting_impulse is necesarry."""
        if self.goal_speed == 0 and speed > 0:
            self.impulse = impulse_dist

        # if 0 < speed < 8.25:
        #     speed = 8.25
        if 0 < speed < 8.25:
            speed = 8.25

        self.goal_speed = speed

    def motor_step(self, goal_speed):
        "High-level main motor control."
        if abs(goal_speed) > 0.01:
            self.motor_set(goal_speed)
        else:
            self.motor_set(0)

    def motor_set(self, motor_pwm_val):
        """low level main motor control"""
        if (motor_pwm_val > 0):
            if (motor_pwm_val > MOTOR_PWM_MAX):
                motor_pwm_val = MOTOR_PWM_MAX
            pin_motor_fwd.ChangeDutyCycle(motor_pwm_val)
            pin_motor_bwd.ChangeDutyCycle(0)
        elif (motor_pwm_val < 0):
            if (motor_pwm_val < -MOTOR_PWM_MAX):
                motor_pwm_val = -MOTOR_PWM_MAX
            pin_motor_fwd.ChangeDutyCycle(0)
            pin_motor_bwd.ChangeDutyCycle(-motor_pwm_val)
        else:
            pin_motor_fwd.ChangeDutyCycle(0)
            pin_motor_bwd.ChangeDutyCycle(0)

    def steer_goal_set(self, angle):
        """Set New goal for steering."""
        if (angle > STEER_MAX_ANGLE):
            angle = STEER_MAX_ANGLE
        elif (angle < STEER_MIN_ANGLE):
            angle = STEER_MIN_ANGLE

        self.goal_steer_angle = angle

    def read_steer_angle(self):
        return (STEER_MIN_ANGLE +
                (STEER_MIN - adc.read_adc(0, gain=ADC_GAIN) + 0.0) /
                (STEER_MIN - STEER_MAX) * (STEER_MAX_ANGLE - STEER_MIN_ANGLE))

    def steer_step(self, goal_steer_angle):
        """high-level steering servo control (PID control)"""

        # steer pid controller parameters
        steer_kp = 0.1
        steer_ki = 0.01
        steer_kd = 0.0
        steer_es = 0
        steer_es_old = 0
        steer_e_old = 0
        steer_e = 100
        v_steer = 0

        steer_e_old = steer_e
        steer_es_old = steer_es

        steer_e = goal_steer_angle - self.read_steer_angle()     # error

        if (abs(steer_e) < 0.5):
            v_steer = 0
            steer_es = 0
        else:
            steer_es = steer_es_old + steer_e  # error sum
            # speed of control signal change
            v_steer = steer_kp * steer_e + steer_ki * steer_es_old + steer_kd * (steer_e - steer_e_old)

        self.steer_set(v_steer)

    def steer_set(self, steer_pwm_val):
        """Low level steering servo control."""

        if (steer_pwm_val < -STEER_PWM_MAX):
            steer_pwm_val = -STEER_PWM_MAX
        elif (steer_pwm_val > STEER_PWM_MAX):
            steer_pwm_val = STEER_PWM_MAX

        pwm = abs(steer_pwm_val) + STEER_PWM_T

        if (steer_pwm_val > 0):
            pin_steer_left.ChangeDutyCycle(0)
            pin_steer_right.ChangeDutyCycle(pwm)
            # print("----", pwm)
        elif (steer_pwm_val < 0):
            pin_steer_left.ChangeDutyCycle(pwm)
            pin_steer_right.ChangeDutyCycle(0)
            # print(pwm,"----")
        else:
            pin_steer_left.ChangeDutyCycle(0)
            pin_steer_right.ChangeDutyCycle(0)
            # print("----","----")


# Initialize DriverController class
Driver = DriverController()
speed_goal_set = Driver.speed_goal_set
steer_goal_set = Driver.steer_goal_set
halt = Driver.halt
start = Driver.start


class myThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        logging.info("Starting: " + self.name)
        Driver.act_loop()
        logging.info("Stopping: " + self.name)


thread1 = myThread(1, "Actuator loop")
thread1.start()
