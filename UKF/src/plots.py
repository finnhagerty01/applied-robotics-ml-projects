import matplotlib.pyplot as plt
import numpy as np
from src.step import load_landmark_pos

'''
This is all just plotting stuff. Pretty self explanatory I think? I was originally plotting theta, but decided to leave it out b/c I could just use innovation. Hence
The commented out code.
'''

def plot_path(states, t_vec, title, output):

    x_pos, y_pos, theta = zip(*states)
    
    fig, (ax0) = plt.subplots()#2, 1, layout='constrained'
    ax0.plot(x_pos, y_pos)
    ax0.quiver(x_pos, y_pos, 1, 0, angles = theta, pivot = 'tip', scale = 50)
    #ax1.plot(t_vec, theta)
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    #ax1.set_xlabel('Time (s)')
    #ax1.set_ylabel('Theta (rad)')
    fig.suptitle(title)

    plt.savefig(output)
    plt.show()

def plot_path_3(dead_state, truth, t_vec, title, output):

    x_pos, y_pos, theta = zip(*dead_state)
    x_true, y_true, theta_true, ground_time = zip(*truth)

    # x_pos_dx = np.diff(x_pos)
    # y_pos_dy = np.diff(y_pos)

    # x_true_dx = np.diff(x_true)
    # y_true_dy = np.diff(y_true)

    theta_deg = np.degrees(theta)
    theta_true_deg = np.degrees(theta_true)

    fig, (ax0) = plt.subplots() #2, 1, layout='constrained'
    ax0.plot(x_pos, y_pos, color = 'r')
    ax0.plot(x_true, y_true, color = 'k')
    ax0.quiver(x_true[::500], y_true[::500],1, 0, pivot = 'tip', angles = theta_true_deg[::500], scale = 50)
    ax0.quiver(x_pos[::50], y_pos[::50], 1, 0, color = 'red', pivot = 'tip', angles = theta_deg[::50], scale = 50)
    ax0.set_xlim(-10, 10)
    ax0.set_ylim(-10, 10)
    # ax1.plot(t_vec, theta, color = 'r')
    # ax1.plot(ground_time, theta_true, color = 'k')
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Theta (rad)')
    fig.legend(['Dead-Reckoned Path', 'Ground Truth'])
    fig.suptitle(title)

    plt.savefig(output)
    plt.show()

def plot_pos_diff(diff, time, output):
    fig = plt.figure()
    plt.plot(time[1:-1], diff)
    plt.title('Position Difference vs Time')
    plt.xlabel('time (s)')
    plt.ylabel('Positional Difference')

    plt.savefig(output)
    plt.show()

def plot_path_UKF(dead_state, truth, t_vec, title, output):

    x_pos, y_pos, theta = zip(*dead_state)
    x_true, y_true, theta_true, ground_time = truth['x_pos'], truth['y_pos'], truth['angle'], truth['time']
    landmarks = load_landmark_pos('data/ds1/ds1_Landmark_Groundtruth.dat')
    landmark_x = landmarks['x [m]']
    landmark_y = landmarks['y [m]']

    # x_pos_dx = np.diff(x_pos)
    # y_pos_dy = np.diff(y_pos)

    # x_true_dx = np.diff(x_true)
    # y_true_dy = np.diff(y_true)

    theta_deg = np.degrees(theta)
    theta_true_deg = np.degrees(theta_true)

    fig, (ax0) = plt.subplots() #2, 1, layout='constrained'
    ax0.plot(x_pos, y_pos, color = 'r')
    ax0.plot(x_true, y_true, color = 'k')
    ax0.quiver(x_true[::500], y_true[::500],1, 0, pivot = 'tip', angles = theta_true_deg[::500], scale = 60)
    ax0.quiver(x_pos[::50], y_pos[::50], 1, 0, color = 'red', angles = theta_deg[::50], pivot = 'tip', scale = 70)
    ax0.set_xlim(-10, 10)
    ax0.set_ylim(-10, 10)
    # ax1.plot(t_vec, theta, color = 'r')
    # ax1.plot(ground_time, theta_true, color = 'k')
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Theta (rad)')
    ax0.scatter(landmark_x, landmark_y, marker='o', s = 20)
    fig.legend(['UKF-Corrected Path', 'Ground Truth'])
    fig.suptitle(title)

    plt.savefig(output)
    plt.show()

def plot_pos_diff_UKF(diff, time, output):
    fig = plt.figure()
    plt.plot(time[0:-2], diff)
    plt.title('Position Difference vs Time for UKF')
    plt.xlabel('time (s)')
    plt.ylabel('Positional Difference (m)')

    plt.savefig(output)
    plt.show()

def plot_innovation(track_innov, output, other_output):
    innov, dt, time_ = zip(*track_innov)
    innov = np.stack(innov, axis = 0)

    fig = plt.figure()
    plt.plot(time_, innov[:, 0])
    plt.title('Innovation of range vs time')
    plt.xlabel('time (s)')
    plt.ylabel('Innovation')

    plt.savefig(output)
    plt.show()

    fig2 = plt.figure()
    plt.plot(time_, innov[:, 1])
    plt.title('Innovation of bearing vs time')
    plt.xlabel('time (s)')
    plt.ylabel('Innovation')

    plt.savefig(other_output)
    plt.show()