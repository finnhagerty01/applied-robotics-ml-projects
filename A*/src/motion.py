import numpy as np

'''
My motion model math <3
'''
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def arc_step(state, control, dt):
    x, y, theta = state
    v, omega = control

    if abs(omega) < 1e-6:
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta
    
    else:
        x_new = x + (v/omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        y_new = y - (v/omega) * (np.cos(theta + omega * dt) - np.cos(theta))
        theta_new = wrap_angle(theta + omega * dt)
    
    return np.array([x_new, y_new, theta_new])