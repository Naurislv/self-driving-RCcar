"""Controls for roomba model car motor/steering actuators (from Raspberry Pi)."""

# Standard imports
import logging

# Local imports
import create

ROOMBA_PORT = '/dev/serial0'


class DriverController(object):
    """Controller/Driver class which is specificly written to control specific hardware."""

    def __init__(self):
        self.robot = None

        self.current_steer = 0
        self.current_speed = 0

    def start(self):
        """Create roomba object with specified arguments."""
        logging.info("Starting roomba controller")

        self.robot = create.Create(ROOMBA_PORT)  # init class
        self.robot.toSafeMode()  # set mode
        self.robot.resetPose()  # reset coordinate system to 0

    def halt(self):
        """Shuw down roomba controller."""
        self.robot.close()

    def steer_goal_set(self, steer):
        """Set roomba current steering angle."""
        self.current_steer = steer * 18
        self.robot.go(self.current_speed, self.current_steer)

    def speed_goal_set(self, speed):
        """Set roomba current speed."""

        self.current_speed = speed
        self.robot.go(self.current_speed, self.current_steer)


# Initialize DriverController class
Driver = DriverController()
speed_goal_set = Driver.speed_goal_set
steer_goal_set = Driver.steer_goal_set
halt = Driver.halt
start = Driver.start
