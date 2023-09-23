"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
import math

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import reactive_obst_avoid
from control import potential_field_control
from control import gradient_attractif_calculation


SCORE_MIN = 5

# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with display
        self.counter = 0

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        
        # if the robot is following the wall
        self.following_wall = False
        
        # Path between the actual position and the goal
        self.path = []
        self.path_copy = []
        self.copied = False
        self.iterator_path = 0
        self.point_path = [0, 0, 0]
    
    def follow_line(self, pose, goal_point):
        """
        Follow the path to the goal after the A*
        """
        # Convert from map coordinates to world coordinates of the goal point
        x_goal_point, y_goal_point = self.tiny_slam._conv_map_to_world(goal_point[0], goal_point[1])
        goal_point = [x_goal_point, y_goal_point, 0]
        
        
        ######## First method as an attempt to make the robot return to its origin (with an iterator) ########
        
        # When the distance between the robot and the goal is less than 10,
        # we consider that the robot has reached the goal, so we calculate
        # the attractive gradient for another point along the way.
        # To move from one point to another, the iterator "iterator_path" is used.
        # In this case, the robot walks every 10 points.
        if self.tiny_slam.heuristic(pose, goal_point) < 10:
            self.iterator_path += 10
            if self.iterator_path < len(self.path):
                self.point_path = self.path[self.iterator_path]
        gradient = gradient_attractif_calculation(pose, self.point_path)
        
        #####################################################################################################
        
        
        
        ######## Second method as an attempt to make the robot return to its origin (with a copy of the path) ########
        '''
        gradient = gradient_attractif_calculation(pose, goal_point)
        '''
        #####################################################################################################
        
        
        
        # Gradient normalization (regardless of which method is used)
        norm_gradient = np.linalg.norm(gradient)

        # Angle calculation from the attractive gradient
        angle = (math.atan2(gradient[1], gradient[0]) - pose[2]) / (2*math.pi)
        command = {"forward": 0.1*norm_gradient, "rotation": angle}
        
        return command


    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1
        
        # During the first 300 moments, the robot moves doing the cartography.
        # It can move from reacting to obstacles or from potential field control (attractive and repulsive potential).
        if self.counter < 300:
            
            # Compute new command speed to perform obstacle avoidance
            command, self.following_wall = reactive_obst_avoid(self.lidar(), self.following_wall)
            
            # Compute new command with the potential field control
            '''
            goal = [-50, -200, 1]
            command = potential_field_control(self.lidar(), self.odometer_values(), goal)
            '''
        
        # After 300 counts, the robot is instructed to stop.
        # This command seeks to stop the robot a little before calculating the path,
        # since the robot model has inertia.
        elif self.counter < 350:
            command = {"forward": 0, "rotation": 0}
        
        # After 350 counts, the path is obtained, more specifically when the counter is 400.
        # When there is a path, the robot follows that path to return to its original point.
        else:
            command = {"forward": 0, "rotation": 0}
            # Calculate the path
            if self.counter == 400:
                self.path = self.tiny_slam.plan(self.tiny_slam.get_corrected_pose(self.odometer_values()), self.corrected_pose)
            
            
            # Two different methods were implemented to try to move the robot to its original position.
            
            ######## First method as an attempt to make the robot return to its origin (with an iterator) ########
            
            if len(self.path) != 0:
                command = self.follow_line(self.odometer_values(), self.path[self.iterator_path])
            
            #####################################################################################################
            
            
            
            ######## Second method as an attempt to make the robot return to its origin (with a copy of the path) ########
            
            # Copy the path only once during the entire simulation
            '''
            if len(self.path) != 0:
                if not self.copied:
                    self.path_copy = self.path.copy()
                    self.copied = True
            '''
                    
            # The robot must return to the first path_copy point.
            # If the distance between the robot and the first point is small,
            # we consider that the robot has reached this first point.
            # So the first point of the path_copy becomes the point that is 40 points ahead.
            '''
            if len(self.path_copy) != 0:
                command = self.follow_line(self.odometer_values(), self.path_copy[0])
                if self.tiny_slam.heuristic(self.odometer_values(), self.path_copy[0]) < 20:
                    self.path_copy = self.path_copy[41:]
            '''
            ##############################################################################################################    
                
                
            ######## Other command possibilities, if we are not interested in calculating a path and returning the robot to the initial position. ########    
            
            # The robot just keeps walking and doing the cartography.
            '''
            command, self.following_wall = reactive_obst_avoid(self.lidar(), self.following_wall)
            '''
            
            # The robot stops forever.
            '''
            command = {"forward": 0, "rotation": 0}
            '''
            ##############################################################################################################################################
            
        
        # Calculates the score and updates the map with the improved cartography after the score is calculated.
        score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
        self.tiny_slam.update_map(self.lidar(), self.odometer_values(), self.path)
        '''
        if score > SCORE_MIN:
            self.tiny_slam.update_map(self.lidar(), self.odometer_values(), self.path)
        '''
            
        return command
