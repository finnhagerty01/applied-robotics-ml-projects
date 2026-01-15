import numpy as np

def heur(start, goal) -> float:
    
    init_y, init_x = start
    goal_y, goal_x = goal

    h = max([abs(init_y - goal_y), abs(init_x - goal_x)])

    return h

def euc_heur(start, goal) -> float:
    init_y, init_x = start
    goal_y, goal_x = goal
    
    h = np.sqrt((init_y - goal_y)**2 + (init_x - goal_x)**2)

    return h