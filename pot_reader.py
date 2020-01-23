# !/usr/bin/env python
from pydub import AudioSegment
from pydub.playback import play
import rospy
from std_msgs.msg import Float64

def callback(data):
    rospy.loginfo(data.data)
    if data.data > 100:
    	song = AudioSegment.from_wav("danger.wav")
    	play(song)
    	return 


def listener():
    rospy.init_node('pot_listener', anonymous=True)
    rospy.Subscriber('chatter', Float64, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
