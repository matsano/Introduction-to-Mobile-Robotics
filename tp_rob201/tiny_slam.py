""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle
import math
import numpy

import cv2
import numpy as np
from matplotlib import pyplot as plt

MAX_PROB = 0.95
MIN_PROB = -0.95

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map *  self.resolution
        y_world = self.y_min_world + y_map *  self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val


    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        score = 0
        # Estimer les positions des detections du laser dans le repere global
        distance_obst = lidar.get_sensor_values()
        direction_obst = lidar.get_ray_angles()
        x = distance_obst * numpy.cos(direction_obst + pose[2]) + pose[0]
        y = distance_obst * numpy.sin(direction_obst + pose[2]) + pose[1]
        
        # Supprimer les points qui ne correspondent pas à des obstacles
        x = x[distance_obst < lidar.max_range]
        y = y[distance_obst < lidar.max_range]
        
        # Convertir ces positions dans le repere de la carte
        x_map, y_map = self._conv_world_to_map(x, y)
        
        # Supprimer les points hors de la carte
        x_inside_map = x_map >= 0
        y_inside_map = y_map >= 0
        ## Produit booleen (AND)
        points_inside_map = x_inside_map * y_inside_map
        x_map = x_map[points_inside_map]
        y_map = y_map[points_inside_map]
        
        x_inside_map = x_map < self.x_max_map
        y_inside_map = y_map < self.y_max_map
        points_inside_map = x_inside_map * y_inside_map
        x_map = x_map[points_inside_map]
        y_map = y_map[points_inside_map]
        
        # Additionner les valeurs dans la carte et calculer le score
        score = numpy.sum(self.occupancy_map[x_map, y_map])

        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if (odom_pose_ref is None):
            odom_pose_ref = self.odom_pose_ref
        
        d0 = math.sqrt((odom[0] - odom_pose_ref[0])**2 + (odom[1] - odom_pose_ref[1])**2)
        # sum_angle = theta0_ref + alpha0
        sum_angle = math.atan2((odom[1]-odom_pose_ref[1]), (odom[0]-odom_pose_ref[0]))
        x = odom_pose_ref[0] + d0 * math.cos(sum_angle)
        y = odom_pose_ref[1] + d0 * math.sin(sum_angle)
        theta = odom_pose_ref[2] + odom[2]
        odom_pose = numpy.array([x, y, theta])
        corrected_pose = odom_pose

        return corrected_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        best_score = 0
        best_odom = self.odom_pose_ref
        
        # Calculer le score avec la position de reference de l'odometrie
        best_score = self.score(lidar, best_odom)
        
        # Trouver le meilleur score
        N = 100
        n_repetition = 0
        while n_repetition < N:
            # Tirer un offset et ajoutez-le à la position de reference de l'odometrie
            offset = []
            offset.append(numpy.random.normal(0.0, 4.0))
            offset.append(numpy.random.normal(0.0, 4.0))
            offset.append(numpy.random.normal(0.0, 0.5))
            new_odom_ref = best_odom + offset
            
            # Calculer le score avec cette nouvelle position de ref
            new_odom = self.get_corrected_pose(odom, new_odom_ref)
            score = self.score(lidar, new_odom)
            
            # Memoriser le meilleur score
            if score > best_score:
                best_score = score
                best_odom = odom
                n_repetition = 0
            else:
                n_repetition += 1
        
        # Mettre à jour la position de ref
        self.odom_pose_ref = best_odom
        

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        # FALTA CALCULAR AS PROBABILIDADES (PERGUNTAR PARA O PROF)
        
        # Conversion des coordonnees polaires en cartesiennes
        r = lidar.get_sensor_values()
        angle = lidar.get_ray_angles()
        x = r * numpy.cos(angle + pose[2]) + pose[0]
        y = r * numpy.sin(angle + pose[2]) + pose[1]

        # Detection des obstacles
        x_obs = x[r < lidar.max_range - 20]
        y_obs = y[r < lidar.max_range - 20]
        
        self.add_map_points(x_obs, y_obs, 0.5)
        for i in range (numpy.size(x)):

            # if r[i] < lidar.max_range - 20:
                # self.add_map_points(x[i], y[i], 0.5)
            self.add_map_line(pose[0], pose[1], x[i], y[i], -0.1)
        
        # Seuillage des probabilites
        self.occupancy_map[self.occupancy_map > MAX_PROB] = MAX_PROB
        self.occupancy_map[self.occupancy_map < MIN_PROB] = MIN_PROB
        
        #self.display(pose)
        self.display2(pose)

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """
        # TODO for TP5

        path = [start, goal]  # list of poses
        return path

    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world}, fid)

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi,np.pi,np.pi/1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x,pt_y])
