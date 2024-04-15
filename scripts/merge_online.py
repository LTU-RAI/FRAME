#! /usr/bin/env python3

""" 
	FRAME: Fast and Robust Autonomous 3D point cloud Map-merging
	 			for Egocentric multi-robot exploration 
"""

import numpy as np
import open3d as o3d
import time, rospy, yaml
from typing import List
import rospkg, pygicp
from sklearn.neighbors import KDTree
from sensor_msgs.msg import PointCloud2
from utils_frame.visualize import visualize
from utils_frame.pcd_publisher import pcd_publisher
from utils_frame.robotFeatures import RobotFeatures


## Get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()
## Get the file path for rospy_tutorials
rospack_path = rospack.get_path('frame')
## Get config file path
config_path = rospack.get_path('frame') + '/config/params.yaml'

## Load parameters
params = yaml.load(open(config_path), Loader=yaml.FullLoader)
## Set parameters
voxel_down_sample = params['voxel_down_sample']
sphere_radius = params['sphere_radius']
filtering_radius = params['filtering_radius']
nb_neighbors = params['nb_neighbors']
std_ratio = params['std_ratio']
frame_id = params['frame_id']
max_cor_dist_1 = params['max_cor_dist_1']
max_cor_dist_2 = params['max_cor_dist_2']	
m1_topic = params['r1_topic']
m2_topic = params['r2_topic']
r1_ns = params['r1_ns']
r2_ns = params['r2_ns']
merged_topic = params['merged_topic']
num_threads = params['num_threads']
rate = params['rate']

## Target vectors, poses and map path
tr1_topic = "/tr1"
v1_topic =  "/v1"
## Incoming vectors, poses and map path
tr2_topic = "/tr2"
v2_topic = "/v2"


class Node:
	def __init__(self, robot1:RobotFeatures, robot2:RobotFeatures, 
					sample_radius:float=7.5, num_threads:int=8, 
						max_cor_dist_1:int=3, max_cor_dist_2:int=1) -> None:
		## Initialize two agens
		self.r1 = robot1
		self.r2 = robot2
		## Initialize merged map
		self.merged_map = o3d.geometry.PointCloud()
		## Initialize merged map publisher
		self.m_publisher = rospy.Publisher(merged_topic, PointCloud2, queue_size=1, latch=True)
		## Set parameters
		self.sample_radius = sample_radius
		self.num_threads = num_threads
		self.max_cor_dist_1 = max_cor_dist_1
		self.max_cor_dist_2 = max_cor_dist_2

	def extract_sphere(self, pcd:o3d.geometry.PointCloud, 
						center:float, radius:float) -> o3d.geometry.PointCloud:
		## Convert the o3d point cloud to a numpy array
		points = np.asarray(pcd.points)
		## Calculate the distance between each point and the center of the sphere
		distances = np.linalg.norm(points - center, axis=1)
		## Select the points that are within the sphere
		inside_sphere = points[distances <= radius]
		## Create a new point cloud from the selected points
		sampled_pcd = o3d.geometry.PointCloud()
		sampled_pcd.points = o3d.utility.Vector3dVector(inside_sphere)
		return sampled_pcd

	def merge(self) -> List[List[float]]:
		## Wait until the maps are not empty
		while np.shape(np.asarray(self.r1.map.points))[0] == 0 and np.shape(np.asarray(self.r1.map.points))[0] == 0:
			rospy.loginfo("Waiting to receive maps...")
			rospy.sleep(1)
		## Receive maps
		rospy.loginfo("Received maps...")
		## Print size of maps
		rospy.loginfo("Number of points in r1/map: %.f" % np.shape(np.asarray(self.r1.map.points))[0])
		rospy.loginfo("Number of points in r2/map: %.f" % np.shape(np.asarray(self.r2.map.points))[0])
		## Start timer
		start_time = time.time()
		## Make kd-tree with target vectors and query with incoming vectors
		# tree = KDTree(self.r1.q)
		# dist, ind =  tree.query(self.r2.q, k=1)
		## Get indexes for vector pairs with min and max distance
		argmin = 0 # np.argmin(dist)
		argmin_index = 0 # argmin_index
		## Log index info
		rospy.loginfo("Min indexes info: " + str(argmin) + ", " + str(argmin_index))
		## Estimate yaw discrepancy
		# yaw = orientation_estimate(tensorflow.expand_dims((self.r1.w[argmin_index], self.r2.w[argmin]),0)) # ( self.r1.w[argmin_index][0], self.r2.w[argmin])
		dtheta = 0.0 #np.pi - ((np.arccos(yaw[0][0]) + np.arcsin(yaw[0][1])))*np.pi/2 #(1 - (yaw[0][0] + yaw[0][1]))*np.pi/2
		## Log yaw info
		rospy.loginfo("Yaw discrepancy: %.3f deg \n" % (180*dtheta/np.pi))
		## Get points inside the spheres
		tic = time.time()
		self.r1.sphere = self.extract_sphere(self.r1.map.voxel_down_sample(voxel_down_sample), [0.0, 0.0, 0.5], self.sample_radius)
		self.r2.sphere = self.extract_sphere(self.r2.map.voxel_down_sample(voxel_down_sample), [0.0, 0.0, 0.5], self.sample_radius)
		toc = time.time()
		## Log time info
		rospy.loginfo("Sampling took %.3f seconds \n" % (toc - tic))
		## Create initial transformation
		T1 = [[np.cos(0), -np.sin(0), 0, 0.0], \
				[np.sin(0), np.cos(0), 0, 0.0], \
				[0, 0, 1, -0.5], \
				[0, 0, 0, 1]]
		T2 = [[np.cos(dtheta), -np.sin(dtheta), 0, 0], \
				[np.sin(dtheta), np.cos(dtheta), 0, 0], \
				[0, 0, 1, 0], \
				[0, 0, 0, 1]]
		T3 = [[np.cos(0), -np.sin(0), 0, 0.0], \
				[np.sin(0), np.cos(0), 0, 0.0], \
				[0, 0, 1, 0.5], \
				[0, 0, 0, 1]]        
		self.T = np.array(T3) @ np.array(T2) @ np.array(T1)
		## Do transformations
		self.r2.map.transform(self.T)
		self.r2.sphere.transform(self.T)
		## Register centers
		c1_ = [[0.0, 0.0, 0.5]]
		self.r1.center.points = o3d.utility.Vector3dVector(np.asarray(c1_))
		c2_ = [[0.0, 0.0, 0.5]]
		self.r2.center.points = o3d.utility.Vector3dVector(np.asarray(c2_))
		## Log initial transform
		rospy.loginfo("Estimated initial transform: \n{} \n".format(self.T))
		## Perform Fast GICP
		inter_guess = pygicp.align_points(np.asarray(self.r1.sphere.points),
										np.asarray(self.r2.sphere.points),
										method='GICP', max_correspondence_distance=self.max_cor_dist_1, 
										num_threads=self.num_threads)
		gicp_result = pygicp.align_points(pygicp.downsample(np.asarray(self.r1.sphere.points), 0.75),\
										pygicp.downsample(np.asarray(self.r2.sphere.points), 0.75), \
										initial_guess=inter_guess, method='VGICP', 
										max_correspondence_distance=self.max_cor_dist_2,
										num_threads=self.num_threads)
		## Log result transformation matrix 
		rospy.loginfo("GICP result transform: \n{} \n".format(gicp_result))
		## Transform incoming map
		self.r2.map.transform(gicp_result)
		self.r2.sphere.transform(gicp_result)
		## End timer4
		end_time = time.time()
		## Log time info
		rospy.loginfo("Merging took %.3f seconds \n" % (end_time - start_time))
		## Final transform
		self.final_T = gicp_result @ self.T
		self.r2.center.transform(self.final_T)    
		## Log final transformation matrix 
		rospy.loginfo("Final transform: \n{} \n".format(self.final_T))
		## Concatinate maps to create merged map, each is a Nx3 numpy array
		self.merged_map.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.r1.map.points), 
																	  np.asarray(self.r2.map.points)), axis=0))
		## Publish merged map
		pcd_publisher(frame_id, self.m_publisher, self.merged_map.points)
		## Return final Transform
		return self.final_T

	# def publish_maps(self) -> None:
	# 	## Publishers
	# 	publishers = [self.r1.m_publisher, self.r2.m_publisher]
	# 				#   self.r1.s_publisher, self.r2.s_publisher, 
	# 				#   self.r1.c_publisher, self.r2.c_publisher]
	# 	## Point clouds
	# 	self.point_clouds = [self.r1.map, self.r2.map]
	# 						#  self.r1.sphere, self.r2.sphere, 
	# 						#  self.r1.center, self.r2.center]
	# 	## Publish all maps and spheres
	# 	for publisher, pcd in zip(publishers, self.point_clouds):
	# 		pcd_publisher(frame_id, publisher, pcd.points)


if __name__ == '__main__':
	## Initialize ROS node
	rospy.init_node('map_merge', anonymous=True)
	## Set node working rate
	ros_rate = rospy.Rate(rate) 
    ## Load incoming and target maps
	## Create object
	robot1 = RobotFeatures(m1_topic, from_file=False, namespace=r1_ns)
	robot2 = RobotFeatures(m2_topic, from_file=False, namespace=r2_ns)
	node = Node(robot1, robot2, 
			sample_radius=sphere_radius,
			max_cor_dist_1=max_cor_dist_1,
			max_cor_dist_2=max_cor_dist_2, 
			num_threads=num_threads)
	T = node.merge()
	## Keep nodes running
	while not rospy.is_shutdown():
		rospy.spin()