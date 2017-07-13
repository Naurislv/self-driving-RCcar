import create
import logging
import threading

ROOMBA_PORT = '/dev/serial0'


class DriverController(object):

    def __init__(self):
        self.robot = create.Create(ROOMBA_PORT)  # init class
        self.robot.toSafeMode()  # set mode
        self.robot.resetPose()  # reset coordinate system to 0

    def act_loop(self):
        # param0 : cm / s
        # param1 : degree / s

        logging.info('act_loop initialized')
        speed = 10.0
        angle = 10.0
        print("GOOOO")
        self.robot.go(speed, angle)

    def start(self):
        print('Starting')
        logging.info("starting")

    def halt(self):
        self.robot.close()

    def steer_goal_set(self):
        pass

    def speed_goal_set(self):
        pass


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
