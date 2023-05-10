import math
import numpy


""" A set of robotics control functions """

def wall_found(dist_straight):
    '''
    Identifier si un mur a été trouvé
    '''
    if numpy.min(dist_straight) < 40:
        return True
    else:
        return False
    

def go_to_wall(lidar):
    '''
    Go to a wall
    '''
    direction_obst = lidar.get_ray_angles()
    distance_obst = lidar.get_sensor_values()
    
    # Considerer les obstacles situés devant sous un angle de pi/3 rad
    index_straight = numpy.abs(direction_obst) < numpy.pi/6
    straight = distance_obst[index_straight]
    
    # Considerer les obstacles situés à droite sous un angle de pi/5 rad
    index_right = numpy.abs(direction_obst+numpy.pi/2) < numpy.pi/10
    right = distance_obst[index_right]
    
    # Trouver le obstacle (wall)
    # Si le mur n'est pas trouvé, allez tout droit.
    if not wall_found(straight):
        return {"forward": 0.1, "rotation": 0}, False
    else:
        # Si le robot a atteint un mur,
        # vérifier s'il y a aussi un mur à droite du robot.
        # Sinon, tourner le robot.
        if numpy.min(right) > 20:
            return {"forward": 0, "rotation": 0.1}, False
        else:
            return {"forward": 0, "rotation": 0}, True

def follow_wall(lidar):
    '''
    Follow a wall once the robot has found the wall
    '''
    direction_obst = lidar.get_ray_angles()
    distance_obst = lidar.get_sensor_values()
    
    # Considerer les obstacles situés devant sous un angle de pi/3 rad
    index_straight = numpy.abs(direction_obst) < numpy.pi/6
    straight = distance_obst[index_straight]
    
    # Considerer les obstacles situés à droite sous un angle de pi/4 rad
    index_right = numpy.abs(direction_obst+numpy.pi/2) < numpy.pi/8
    right = distance_obst[index_right]
    
    # Le robot reste à une vitesse de 0.5 et un angle de -0.2 par défault
    displacement = 0.5
    angle = -0.2
    
    # Si un mur se trouve devant, le robot tourne
    if numpy.min(straight) < 30:
        angle = 0.3
        displacement = 0
    # Comme le robot tourne à un angle de -0.2 par défaut,
    # il a besoin d'un angle qui varie entre -0.2 et 0.2 pour contrôler l'approche du robot vers le mur à sa droite.
    elif numpy.min(right) < 30:
        angle = -0.01 * numpy.min(right)
        angle = numpy.clip(angle, -0.2, 0.2)
        displacement = 0.05
    
    return {"forward": displacement, "rotation": angle}


def reactive_obst_avoid(lidar, following_wall):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    
    # is_following == True, the robot follows the wall
    # is_following == True, the robot still does not follow the wall (walks straight towards a wall)
    is_following = following_wall
    
    # Just testing the robot movement
    '''
    # Tourner d'un angle aleatoire quand un obstacle est present
    if numpy.amin(distance_obst) < 50:
        angle = numpy.random.rand()
    else:
        angle = 0
    '''
    
    
    ######## First method for the robot to avoid obstacles (Wall Follower) ########
    '''
    # WALL FOLLOWER
    if following_wall:
        command = follow_wall(lidar)
        is_following = True
    else:
        command, is_following = go_to_wall(lidar)
    '''
    ##############################################################################
    
    
    ######## Second method for the robot to avoid obstacles (Turn in the direction where the distance is greater) ########
    
    # Considerer les obstacles situés dans un angle de pi rad
    direction_obst = lidar.get_ray_angles()
    distance_obst = lidar.get_sensor_values()
    field_vision = (direction_obst < math.pi/2) * (direction_obst > -math.pi/2)
    direction_obst = direction_obst[field_vision]
    distance_obst = distance_obst[field_vision]
    
    # Tourner dans le sens où la distance est plus grande
    min_dist = numpy.amin(distance_obst)
    if min_dist < 15:
        index_min = numpy.where(distance_obst == min_dist)
        angle = -direction_obst[index_min] / (2*math.pi)
    else:
        max_dist = numpy.amax(distance_obst)
        index = numpy.where(distance_obst == max_dist)
        angle = direction_obst[index] / (2*math.pi)
    
    command = {"forward": 0.13, "rotation": angle}
    
    #######################################################################################################################
    
    
    return command, is_following



def gradient_attractif_calculation(pose, goal):
    """
    Calcul du gradient de potentiel attractif
    """
    dist_goal = math.sqrt((goal[0] - pose[0])**2 + (goal[1] - pose[1])**2)
    
    ## Lisser les deplacements du robot
    if dist_goal > 20:
        # Potentiel attractif conique
        K_goal = 1
        gradient_attractif = K_goal * (goal[:2] - pose[:2]) / dist_goal
    else:
        # Potentiel attractif quadratique
        K_goal = 0.1
        gradient_attractif = K_goal * (goal[:2] - pose[:2])
        
    return gradient_attractif
    

def gradient_repulsif_calculation(lidar, pose):
    '''
    Calcul du gradient de potentiel repulsif
    '''
    distances_obs = lidar.get_sensor_values()
    directions_obs = lidar.get_ray_angles()
    # Considerer les obstacles situés dans un angle de pi rad
    field_vision = (directions_obs < math.pi/2) * (directions_obs > -math.pi/2)
    directions_obs = directions_obs[field_vision]
    distances_obs = distances_obs[field_vision]
    
    # Calcul de la position du obstacle
    r_obs = numpy.min(distances_obs)
    index_obs = numpy.where(distances_obs == r_obs)
    angle_obs = directions_obs[index_obs]
    
    # Conversion des coordonnees polaires en cartesiennes
    x = r_obs * numpy.cos(angle_obs + pose[2]) + pose[0]
    y = r_obs * numpy.sin(angle_obs + pose[2]) + pose[1]
    q_obs = numpy.array([x])
    q_obs = numpy.append(q_obs, y) # q_obs = [x, y]
    
    # Calcul du gradient
    K_obs = 10
    d_safe = 20
    if r_obs > d_safe:
        gradient_repulsif = 0
    else:
        gradient_repulsif = (K_obs/(r_obs)**3)*(1/r_obs - 1/d_safe)*(q_obs - pose[:2])
    
    return gradient_repulsif


def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
    # TODO for TP2
    
    # Obtenir gradient de potentiel attractif
    gradient_attractif = gradient_attractif_calculation(pose, goal)
    
    # Obtenir du gradient de potentiel repulsif
    gradient_repulsif = gradient_repulsif_calculation(lidar, pose)
    
    # Gradient total
    gradient_total = gradient_attractif + gradient_repulsif
    norm_gradient = numpy.linalg.norm(gradient_total)
    
    # Calcul d'angle
    angle = (math.atan2(gradient_total[1], gradient_total[0]) - pose[2]) / (2*math.pi)
    
    dist_goal = math.sqrt((goal[0] - pose[0])**2 + (goal[1] - pose[1])**2)
    if dist_goal < 6:
        command = {"forward": 0, "rotation": 0}
        print("ARRIVE")
    else:
        command = {"forward": 0.2*norm_gradient, "rotation": angle}

    return command
