from tof import ToFDriver
import numpy as np
import stretch_body.robot as robot
import time
import argparse

parser = argparse.ArgumentParser(description='Capture input from ToF sensor while closing robot gripper')
parser.add_argument('--tof-port-path', type=str, required=True,
                    help='port path for ToF sensor')
parser.add_argument('--tof-inset', type=int, required=True,
                    help='distance (mm) that ToF sensor is inset from the surface face of the gripper')                 
args = parser.parse_args()


# initialize ToF Sensor
tof_controller = ToFDriver(port_path=args.tof_port_path) # may be inconsistent btw startups, verify port using dmesg

# initialize robot
r = robot.Robot()
r.startup()

# initialize robot wrist for grasping
r.end_of_arm.move_to('reactor_gripper', 2) # fully open gripper
r.end_of_arm.move_to('wrist_pitch', 0)     # pitch to 90 deg, wrist parallel to ground
r.end_of_arm.move_to('wrist_roll', 0)      # roll to 0 deg, gripper fingers in line for grasp
#r.end_of_arm.move_to('wrist_yaw', 0)       # ! may need to change this, yaw is currently not working on my end so I couldn't verify this value

# wait for wrist to finish reaching initial position
time.sleep(2)
# print(r.get_status()['end_of_arm']['wrist_pitch'])

raw_input('Press Enter to close gripper...')

tof_controller.flush()
dist = tof_controller.get_distance()
# dist = 20
while dist > 10:
    # increment the gripper to close
    r.end_of_arm.move_by('reactor_gripper', -0.1)
    dist = tof_controller.get_distance() - args.tof_inset
    print(dist)