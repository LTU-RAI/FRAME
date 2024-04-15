import std_msgs.msg, rospy
import sensor_msgs.point_cloud2 as pcl2

## Publish point cloud
def pcd_publisher(frame_id, pub, pcd) -> None:
    # Create header
    map_header = std_msgs.msg.Header()
    map_header.stamp = rospy.Time.now()
    map_header.frame_id = frame_id
    # Create point cloud
    map_pcd = pcl2.create_cloud_xyz32(map_header, pcd)
    # Publish
    pub.publish(map_pcd)