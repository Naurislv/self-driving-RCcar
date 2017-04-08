"""Controls for Wltoys A969 model car motor/steering actuators (from Raspberry Pi).

Usage:
    import wltoys_a969
    Set speed
    wltoys_a969.speed_goal_set(Speed)    #  Speed in [0..12] (slower..faster)

    Set steering angle
    wltoys_a969.steer_goal_set(Angle)    # Angle in [-18..18] (in degrees)
"""

import RPi.GPIO as IO
import Adafruit_ADS1x15
import time

IO.setwarnings(False)
IO.setmode(IO.BCM)

IO.setup(19, IO.OUT)
pin_motor_fwd = IO.PWM(19, 50)
pin_motor_fwd.start(100)

IO.setup(26, IO.OUT)
pin_steer_right = IO.PWM(26, 100)
pin_steer_right.start(0)

IO.setup(20, IO.OUT)
pin_steer_left = IO.PWM(20, 100)
pin_steer_left.start(0)

adc = Adafruit_ADS1x15.ADS1015()
ADC_GAIN = 1


# parameters
MOTOR_PWM_MAX = 100
MOTOR_PWM_1 = 0
MOTOR_PWM_2 = 12

STEER_MIN = 1123  # Steer servo min feedback value
STEER_MAX = 837  # Steer servo max feedback value
STEER_PWM_T = 25 / 2.55  # Steer servo threshold PWM abs value
STEER_PWM_MAX = (255 - STEER_PWM_T) / 2.55  # Steer servo max PWM abs value
STEER_MIN_ANGLE = -18.0  # Steer servo min angle value
STEER_IDLE_ANGLE = 0.0  # Steer servo idle angle value
STEER_MAX_ANGLE = 18.0  # Steer servo max angle value

# goal values
goal_speed = 0
goal_steer_angle = 0


# goal assign functions
def speed_goal_set(speed):
    """Speed goal set."""
    global goal_speed
    goal_speed = speed
    return


def steer_goal_set(angle):
    """Steer goal set."""
    global goal_steer_angle
    if (angle > STEER_MAX_ANGLE):
        angle = STEER_MAX_ANGLE
    elif (angle < STEER_MIN_ANGLE):
        angle = STEER_MIN_ANGLE
    goal_steer_angle = angle
    return


# low level main motor control
def motor_set(motor_pwm_val):
    """Motor set."""
    if motor_pwm_val > MOTOR_PWM_2:
        motor_pwm_val = MOTOR_PWM_2
    elif motor_pwm_val < MOTOR_PWM_1:
        motor_pwm_val = MOTOR_PWM_1
    pwm = MOTOR_PWM_MAX - motor_pwm_val
    pin_motor_fwd.ChangeDutyCycle(pwm)
    return


# low level steering servo control
def steer_set(steer_pwm_val):
	if (steer_pwm_val < -STEER_PWM_MAX):
		steer_pwm_val = -STEER_PWM_MAX
	elif (steer_pwm_val > STEER_PWM_MAX):
		steer_pwm_val = STEER_PWM_MAX

	pwm = abs(steer_pwm_val) + STEER_PWM_T

	if (steer_pwm_val > 0):
		pin_steer_left.ChangeDutyCycle(0)
		pin_steer_right.ChangeDutyCycle(pwm)
		#print("----", pwm)
	elif (steer_pwm_val < 0):
		pin_steer_left.ChangeDutyCycle(pwm)
		pin_steer_right.ChangeDutyCycle(0)
		#print(pwm,"----")
	else:
		pin_steer_left.ChangeDutyCycle(0)
		pin_steer_right.ChangeDutyCycle(0)
		#print("----","----")


def millis():
	return int(round(time.time() * 1000))

def angle_diff(angle1, angle2):
	a = angle1 - angle2
	if (a > 180.0):
		a = a - 360.0
	elif (a < -180.0):
		a = a + 360.0
	return a


def read_steer_angle():
	return STEER_MIN_ANGLE + (STEER_MIN-adc.read_adc(0, gain=ADC_GAIN)+0.0) / (STEER_MIN-STEER_MAX) * (STEER_MAX_ANGLE-STEER_MIN_ANGLE)


# high-level main motor control
def motor_step():
	if goal_speed > 0.01:
		motor_set(goal_speed)
	else:
		motor_set(0)
	return


# steer pid controller parameters
steer_kp = 0.1
steer_ki = 0.01
steer_kd = 0.0
steer_es = 0
steer_es_old = 0
steer_e_old = 0
steer_e = 100
theta_diff = 0
v_steer = 0
sig_steer = 0
t_prev = 0

# high-level steering servo control (PID control)
def steer_step():
	"""Steer step."""
	global steer_kp
	global steer_ki
	global steer_kd
	global steer_es
	global steer_es_old
	global steer_e_old
	global steer_e
	global theta_diff
	global v_steer
	global sig_steer
	global t_prev

	t = millis()

	steer_e_old  = steer_e
	steer_es_old = steer_es

	steer_e = goal_steer_angle - read_steer_angle()     # error

	if (abs(steer_e)<0.5):
		v_steer = 0
		steer_es = 0
	else:
		steer_es = steer_es_old + steer_e		# error sum
		v_steer = steer_kp * steer_e + steer_ki * steer_es_old + steer_kd * (steer_e-steer_e_old)   	# speed of control signal change

	steer_set(v_steer)

	t_prev = t

	return


def act_loop():
	while (True):
		steer_step()
		motor_step()
		time.sleep(0.0001)
	return
