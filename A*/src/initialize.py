import numpy as np

def load_sg3():
    return [((.5, -1.5), (.5, 1.5)), ((4.5, 3.5), (4.5, -1.5)), ((-.5, 5.5), (1.5, -3.5))]

def load_sg7():
    return [((2.45, -3.55), (.95, -1.55)), ((4.95, -.05), (2.45, .25)), ((-.55, 1.45), (1.95, 3.95))]

def initialize(sg):
    x_range = [-2, 5]
    y_range = [-6, 6]
    start, goal = zip(*sg)

    return x_range, y_range, start, goal

if __name__ == '__main__':
    initialize(load_sg3())