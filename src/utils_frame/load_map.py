#! /usr/bin/env python3

from sensor_msgs.msg import PointCloud2 
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import open3d as o3d
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2

PATH_TO_FILE="/home/niksta/Desktop/temp/arena1.pcd"#"/home/niksta/catkin_ws/src/frame/data/maps/cloudGlobal_husky_2.pcd"
NAMESPACE = ""#"/shafter2"
FRAME_ID = "map"


def load_map(path_to_file:str, map_publisher:rospy.Publisher, frame_id:str="map") -> None:
    map = o3d.io.read_point_cloud(path_to_file)
    map_header = std_msgs.msg.Header()
    map_header.stamp = rospy.Time.now()
    map_header.frame_id = frame_id
    map_pcd = pcl2.create_cloud_xyz32(map_header, np.asarray(map.points))
    map_publisher.publish(map_pcd)
    return

if __name__ == '__main__':
    ## Initialize ROS node
    rospy.init_node('map_publisher', anonymous=True)
    ## Initialize map publisher
    map_publisher = rospy.Publisher(NAMESPACE + "/map", PointCloud2, queue_size=1, latch=True)
    load_map(PATH_TO_FILE, map_publisher, FRAME_ID)
    ## Keep nodes running
    while not rospy.is_shutdown():
        rospy.spin()