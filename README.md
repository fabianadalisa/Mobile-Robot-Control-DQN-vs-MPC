
# Real-Time Control of a Mobile Robot: Deep Q-Learning vs MPC

Design, implementation and comparison of two control strategies for autonomous navigation of a mobile robot in a simulated **warehouse**: a model-free **reinforcement learning** approach (Deep Q-Network) and a model-based **Model Predictive Control** approach.
Developed for the *Complex Project & Process Management* MSc program at the University of Naples Federico II.

## Scenario
- 20×20 grid warehouse with perimeter walls and four static 2×2 obstacles (tables)
- Start: (4, 8), heading π/4. Goal: (15, 18)
- Objective: reach the goal efficiently while avoiding collisions

## Robot model
Unicycle (differential-drive) kinematics, integrated with `ode45`:

ẋ = v·cos(θ)   ẏ = v·sin(θ)   θ̇ = ω

State `x = [x, y, θ]`, input `u = [v, ω]`.

## Strategy 1 – Deep Q-Learning (DQN)
- **Environment:** custom `rlFunctionEnv` (step and reset functions), normalized observations `[x/W, y/H, θ/π]`, Gaussian position noise (σ = 0.01)
- **Actions:** 14 discrete (v, ω) commands: forward, backward, left/right turns at three rates, gentle curves, stop
- **Reward shaping:**
  - step reward `−0.5 + 10·progress − (eᵀQe + aᵀRa)` with Q = diag(0.5, 0.5, 0.1), R = diag(0.01, 0.02)
  - +200 goal reached · −80 collision or out of bounds · −5 × residual distance at timeout
- **Agent:** Double DQN, critic network 3 → 64 → 64 → 14 (ReLU), learning rate 5e-4, γ = 0.99, replay buffer 20,000, minibatch 64, ε-greedy exploration
- **Training:** up to 2,000 episodes × 120 steps; evaluation over 50 simulated episodes (success rate and mean reward)

## Strategy 2 – Linear Time-Varying MPC
- Unicycle model linearized at each step around the current heading (Ts = 0.05 s)
- Prediction horizon: 20 steps, receding-horizon implementation
- Quadratic cost on state error and control effort: Q = diag(10, 10, 0.5), R = 0.1·I, terminal weight Q_f = Q
- Input constraints: |v| ≤ 0.47 m/s, |ω| ≤ 3.3 rad/s
- Obstacle avoidance through a collision penalty on predicted positions
- Solved at each step with `fmincon`

## Benchmark
| Feature | DQN | MPC |
|---|---|---|
| Model required | No (model-free) | Yes (model-based) |
| Constraint handling | Via reward shaping | Explicit in the optimization |
| Robustness to noise | Learned during training | Sensitive to model errors |
| Exploration | Built-in (ε-greedy) | Limited to the prediction horizon |

## Repository structure
| File | Content |
|---|---|
| `q_learning.m` | DQN agent definition, training, evaluation and trajectory plots |
| `my_stepfunction.m` | Environment step: action mapping, dynamics, reward and termination |
| `my_resetfunction.m` | Environment reset: map, obstacles, initial state and goal |
| `my_pitch.m` | Unicycle kinematic model |
| `Mpc_robot.m` | LTV-MPC controller and warehouse simulation |

## How to run
Requirements: MATLAB with **Reinforcement Learning Toolbox**, **Deep Learning Toolbox** and **Optimization Toolbox**.
- DQN: run `q_learning.m`
- MPC: run `Mpc_robot.m`

## Author
Fabiana D'Alisa, MSc student in Management Engineering, University of Naples Federico II
