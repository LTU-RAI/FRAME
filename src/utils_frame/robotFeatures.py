import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from frame.msg import DescriptorsArray
from std_msgs.msg import Float32MultiArray


class RobotFeatures:
    def __init__(self, vectors:str, poses:str, map:str, 
                    from_file:bool=True, namespace:str="", ) -> None:
        ## Initialize map o3d object
        self.map = o3d.geometry.PointCloud()
        ## Initialize center and sphere
        self.sphere = o3d.geometry.PointCloud()
        self.center = o3d.geometry.PointCloud()
        ## Initialize publishers
        self.m_publisher = rospy.Publisher(namespace + "/map", PointCloud2,
                                            queue_size=1, latch=True)
        self.s_publisher = rospy.Publisher(namespace + "/sphere", PointCloud2, 
                                            queue_size=1, latch=True)
        self.c_publisher = rospy.Publisher(namespace + "/center", PointCloud2,
                                            queue_size=1, latch=True)
        
        ## Initialize based on choice of files or topics
        if from_file:
            self.from_file(vectors, poses, map, namespace)
        else:
            self.from_topic(vectors, poses, map, namespace)
 
    def from_file(self, vectors:str, poses:str, map:str, namespace:str) -> None:
        ## Load incoming and target maps
        self.map = o3d.io.read_point_cloud(map)
        rospy.loginfo("Number of points in " + namespace + \
                        "/map: %.f" % np.shape(np.asarray(self.map.points))[0])
        # self.map, _ = self.map.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)	
        ## Load target vectors and poses
        self.vectors = np.load(vectors, allow_pickle=True)
        self.poses = np.load(poses, allow_pickle=True)
        ## Load target vectors
        self.vectors = np.reshape(self.vectors, (2, len(self.vectors), 64))
        ## Load target poses
        self.poses = np.reshape(self.poses, (len(self.poses), 7))
        ## Split to q and w vectors
        self.q, self.w = self.vectors[0], self.vectors[1]

    def from_topic(self, vectors:str, poses:str, map:str, namespace:str):
        ## Subsrcibe to global map
        self.map_subscriber = rospy.Subscriber(namespace + map, \
                                PointCloud2, self.map_callback, queue_size=1)
        ## Subscribe to descriptor vectors
        self.desriptors_subscriber = rospy.Subscriber(namespace + vectors, \
                                        DescriptorsArray, self.descriptors_callback, queue_size=1)
                ## Subscribe to descriptor vectors
        ## Subscribe to poses
        self.poses_subscriber = rospy.Subscriber(namespace + poses, \
                                    Float32MultiArray, self.poses_callback, queue_size=1)

    def map_callback(self, pcd_msg:PointCloud2) -> None:
        ## Read pcd_msg
        map_points = list(point_cloud2.read_points(pcd_msg, 
                            field_names=("x", "y", "z", "intensity"), skip_nans=True))
        ## Make into open3d object
        if map_points != []:
            self.map.points = o3d.utility.Vector3dVector(map_points)
    
    def descriptors_callback(self, decsriptorArray_msg:DescriptorsArray) -> None:
        ## Get q and w vectors
        self.vectors = decsriptorArray_msg.vectors
        self.q = [x.q for x in self.vectors]
        self.w = [y.w for y in self.vectors]

    def poses_callback(self, poses_msg:Float32MultiArray) -> None:
        ## Get poses array
        self.poses = np.reshape(poses_msg.data,(int(len(poses_msg.data)/7), 7))