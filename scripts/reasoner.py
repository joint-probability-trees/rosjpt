#!/usr/bin/env python3
from rosjpt import JPTReasoner
import rospy


if __name__ == "__main__":
    reasoner = JPTReasoner()
    rospy.spin()