import math
import numpy


""" A set of robotics control functions """

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    direction_obst = lidar.get_ray_angles()
    distance_obst = lidar.get_sensor_values()
    
    # Considerer les obstacles situés dans un angle de pi rad
    for direc in direction_obst:
        if abs(direc) > math.pi/2:
            index = numpy.where(direction_obst == direc)
            direction_obst = numpy.delete(direction_obst, index)
            distance_obst = numpy.delete(distance_obst, index)
            
    # Tourner dans le sens où la distance est plus grande
    max_dist = numpy.amax(distance_obst)
    index = numpy.where(distance_obst == max_dist)
    angle = direction_obst[index] / (2*math.pi)
    
    command = {"forward": 0.15, "rotation": angle}

    return command

def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
    # TODO for TP2
    # REPULSIVO + ATRATIVO NN ESTA FUNCIONANDO
    # CHANGER ANGLE my_world !!!!!!
    
    # Calcul du gradient de potentiel attractif
    dist_goal = math.sqrt((goal[0] - pose[0])**2 + (goal[1] - pose[1])**2)
    ## Lisser les deplacements du robot
    if dist_goal > 10:
        K_goal = 1
        gradient_attractif = K_goal * (goal[:2] - pose[:2]) / dist_goal
    else:
        K_goal = 0.5
        gradient_attractif = K_goal * (goal[:2] - pose[:2])
    
    # Calcul du gradient de potentiel repulsif
    ## Calcul de la position du obstacle
    distances_obs = lidar.get_sensor_values()
    directions_obs = lidar.get_ray_angles()
    r_obs = numpy.min(distances_obs)
    index_obs = numpy.where(distances_obs == r_obs)
    angle_obs = directions_obs[index_obs]
    ### Conversion des coordonnees polaires en cartesiennes
    x = r_obs * numpy.cos(angle_obs + pose[2]) + pose[0]
    y = r_obs * numpy.sin(angle_obs + pose[2]) + pose[1]
    q_obs = numpy.array([x])
    q_obs = numpy.append(q_obs, y)
    ## Calcul du gradient
    K_obs = 1
    d_safe = 20
    if r_obs > d_safe:
        gradient_repulsif = 0
    else:
        gradient_repulsif = (K_obs/(r_obs)**3)*(1/r_obs - 1/d_safe)*(q_obs - pose[:2])
    
    # Gradient total
    gradient_total = gradient_attractif + gradient_repulsif
    gradient_total = gradient_attractif
    norm_gradient = numpy.linalg.norm(gradient_total)
    
    ## Calcul d'angle
    angle = (math.atan2(gradient_total[1], gradient_total[0]) - pose[2]) / (2*math.pi)
    
    if dist_goal < 6:
        command = {"forward": 0, "rotation": 0}
        print("ARRIVE")
    else:
        command = {"forward": 0.3*norm_gradient, "rotation": angle}

    return command
