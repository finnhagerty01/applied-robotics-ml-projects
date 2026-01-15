from src.step import make_step2_plot
from src.step import make_step3_plot
from src.step import step6outputs
from src.UKF import initialize
from src.UKF import UKF_simulate
from src.UKF import tuned_initialize
from src.UKF import UKF_tuned_simulate

if __name__ == "__main__":
    make_step2_plot()
    make_step3_plot()
    step6outputs()
    UKF_simulate(.01, 1)
    UKF_simulate(1, 1)
    UKF_simulate(4, 1)
    UKF_simulate(1, .01)
    UKF_simulate(1, 4)
    UKF_tuned_simulate()