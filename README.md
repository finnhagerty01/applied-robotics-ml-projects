# Robotics State Estimation, Planning, and Learning

State estimation, path planning, and learning projects from a graduate robotics AI course, including a UKF, A* search with online replanning, and GMM–GMR–based learning.

---

## Why These Projects Belong Together

These projects cover three core components of autonomous robotic systems:

- **State Estimation** — inferring the robot’s state under uncertainty (Unscented Kalman Filter)
- **Planning & Control** — computing and executing collision-free paths (A* with online replanning)
- **Learning** — modeling structure in data for prediction (Gaussian Mixture Models + Regression)

Together, they reflect a systems-level view of robotics rather than isolated algorithms.

---

## Repository Structure
'''
robotics-state-estimation-planning-learning/
├── UKF/
├── Astar-Heuristic/
├── GMM-GMR/
├── assets/
│ └── figures/
├── data/
└── README.md
'''

Each project folder contains:
- a standalone implementation
- a project-specific README
- scripts for generating plots and evaluation metrics

---

## Projects

### 1. UKF — Unscented Kalman Filter
Nonlinear state estimation using sigma-point propagation under noisy motion and measurement models.

📁 `UKF/`

---

### 2. A* Search + Online Replanning
Grid-based path planning with obstacle inflation, partial observability, and path execution.

📁 `Astar-Heuristic/`

---

### 3. GMM–GMR Learning
Probabilistic learning using Gaussian Mixture Models and Gaussian Mixture Regression.

📁 `GMM-GMR/`

---

## Notes

- Large datasets are not included to keep the repository lightweight.
- Each project README documents expected data formats and outputs.
- Data can be found at http://asrl.utias.utoronto.ca/datasets/mrclam/index.html

---

## License

MIT License
