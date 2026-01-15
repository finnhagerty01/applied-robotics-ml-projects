# A* Search with Online Replanning and Path Execution

## Goal

Plan collision-free paths in a 2D grid world and execute them under realistic constraints.

This project includes:
- Offline A* planning (known map)
- Online A* replanning under partial observability
- Obstacle inflation for robot footprint
- Path execution with heading and acceleration limits

---

## Key Takeaways

- Shows how heuristic search scales with grid resolution
- Demonstrates online replanning as new obstacles are discovered
- Highlights the impact of obstacle inflation on safety vs. path optimality

---

## Example Output

![A* planning and execution](../assets/figures/astar_execution.png)

---

## Design Decisions

- 8-connected grid with admissible heuristic
- Obstacle inflation to approximate robot radius
- Online planner expands nodes based only on locally observed obstacles
- Simple kinematic controller used for path tracking

---

## Outputs

- Planned path visualization
- Occupancy grid (coarse vs. fine)
- Robot pose trace during execution

---

## How to Run

See `src/` for the main planning and execution scripts.
