"""Controls for roomba model car motor/steering actuators (from Raspberry Pi)."""

# Standard imports
import logging

# Local imports
from roomba import create

ROOMBA_PORT = '/dev/serial0'


class DriverController(object):
    """Controller/Driver class which is specificly written to control specific hardware."""

    def __init__(self):
        self.robot = None

        self.current_tcr = 0
        self.current_cmps = 0

    def start(self):
        """Create roomba object with specified arguments."""
        logging.info("Starting roomba controller")

        self.robot = create.Create(ROOMBA_PORT)  # init class
        self.robot.toSafeMode()  # set mode
        self.robot.resetPose()  # reset coordinate system to 0

    def halt(self):
        """Shuw down roomba controller."""
        logging.info("Closing robot")
        self.robot.stop()
        # self.robot.close()

    def steer_goal_set(self, tcr):
        """Set roomba current Degree Per Second.

        Inputs:
            tcr: Turning Circle Radius im mm
        """

        # "-" because clockwise
        # Formula extracrected from robot.go code (Go To Definition to check more)
        # self.current_dps = - math.degrees(10.0 * self.current_cmps / tcr)

        # logging.info("Controls %s %s", self.current_dps, self.current_cmps)
        # self.robot.go(self.current_cmps, self.current_dps)

        self.current_tcr = tcr
        self.robot._drive(10.0 * self.current_cmps, self.current_tcr)  # pylint: disable=W0212

    def speed_goal_set(self, speed):
        """Set roomba current speed - cm per second"""
        self.current_cmps = speed

        # logging.info("Controls %s %s", self.current_dps, self.current_cmps)
        self.robot.go(self.current_cmps, self.current_tcr)


# Initialize DriverController class
DRIVER = DriverController()
SPEED_GOAL_SET = DRIVER.speed_goal_set
STEER_GOAL_SET = DRIVER.steer_goal_set
HALT = DRIVER.halt
START = DRIVER.start
