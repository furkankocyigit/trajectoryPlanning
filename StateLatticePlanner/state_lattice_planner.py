
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__))
                + "/../ModelPredictiveTrajectoryGenerator/")

try:
    import model_predictive_trajectory_generator as planner
    import motion_model
except:
    raise

table_path = os.path.dirname(os.path.abspath(__file__)) + "/lookuptable.csv"


show_animation = True

def search_nearest_one_from_lookuptable(tx, ty, tyaw, lookup_table):
    mind  = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):

        dx   = tx - table[0]
        dy   = ty - table[1]
        dyaw = tyaw - table[2]
        d    = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)

        if d <= mind:
            minid = i
            mind  = d

    return lookup_table[minid]


def get_lookup_table():
    data = pd.read_csv(table_path)

    return np.array(data)

def generate_path(target_states, k0):
    # x, y, yaw, s, km, kf

    lookup_table = get_lookup_table
    result = [] 
    
    for state in target_states:
        bestp = search_nearest_one_from_lookuptable(
            state[0], state[1], state[2], lookup_table)

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [math.sqrt(state[0] ** 2 + state[1] ** 2), bestp[4], bestp[5]]).reshape(3, 1)
        
        x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

        if x is not None:
            print("find goof path")
            result.append(
                [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])
    print("finish path generation")
    return result








def calculate_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max):

    '''
    calculate uniform state

    :param nxy   : number of position sampling
    :param nh    : number of heading sampling
    :param d     : distance of terminal state
    :param a_min : position sampling min angle
    :param a_max : position sampling max angle
    :param p_min : heading sampling min angle
    :param p_max : heading sampling max angle
    :return      : states list
        
    '''
    angle_samples = [i / (nxy - 1) for i in range(nxy)]
    states = sample_states(angle_samples, a_min, a_max, d, p_min, p_max, nh)

    return states


def sample_states(angle_samples, a_min, a_max, d, p_min, p_max, nh):
    states = []
    for i in range(angle_samples):
        a = a_min + (a_max - a_min) * i

        for j in range(nh):
            xf = d * math.cos(a)
            yf = d * math.sin(a)

            if nh == 1:
                yawf = (p_max - p_min) / 2.0 + a
            else:
                yawf = p_min + (p_max - p_min) * j / (nh - 1) + a
            
            states.append([xf, yf, yawf])

    return states

    

def uniform_terminal_state_sampling_test1():

    k0     = 0.0
    nxy    = 5
    nh     = 3
    d      = 20
    a_min  = - np.deg2rad(45.0)
    a_max  =   np.deg2rad(45.0)
    p_min  = - np.deg2rad(45.0)
    p_max  =   np.deg2rad(45.0)
    states =   calculate_uniform_polar_states (nxy, nh, d, a_min, a_max, p_min, p_max)
    result =   generate_path(states, k0)

    for table in result:
        xc, yc, yawc = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        if show_animation:
            plt.plot(xc, yc, "-r")
    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    print("Done")
    

def main():
    uniform_terminal_state_sampling_test1()


if __name__ == '__main__':
    main()