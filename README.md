# AutoCar Exploration Project

## Overview

The AutoCar Exploration Project is designed to develop an autonomous car capable of fully exploring a room. This project is divided into two complementary parts:

1. **Simulation Environment:**  
   A modular, PyBullet-based simulation that provides a virtual room with obstacles and a Husky-based auto car. The simulation is built as an OpenAI Gym environment with a reward function designed to encourage full room coverage.

2. **Physical Car Implementation:**  
   A real-world, Raspberry Pi-based auto car that leverages the control strategies and policies developed in simulation. The physical car integrates sensors, a camera module, and motor controllers to navigate an actual room.

The simulation environment enables rapid prototyping and safe testing of reinforcement learning (RL) algorithms before they are deployed on the physical platform.

## Project Purpose and Goals

- **Full Exploration:**  
  The primary goal is to develop an auto car that can visit all areas of a room. In the simulation, the room is divided into a 100×100 grid (assuming room bounds of –10 to +10 in both x and y).  
  - Entering a new grid cell yields **+100 reward**.
  - Remaining in the same cell incurs a **–5 penalty**.
  - Moving to a previously visited (but different) cell incurs a **–1 penalty**.

- **Smooth, Realistic Control:**  
  Control commands are implemented to ensure smooth acceleration, deceleration, and turning, while avoiding abrupt direction changes. This makes the policies developed in simulation more applicable to the physical car.

- **Seamless Transition from Simulation to Reality:**  
  By sharing control logic between the simulation (AutoCarEnv) and the physical car platform, the project facilitates transferring policies learned in the virtual world to the real-world robot.

## Project Components

### 1. Simulation Environment

- **Environment Setup:**  
  - Ground plane with a floor texture.
  - Surrounding walls and obstacles (e.g., a desk and a chair).
  - A Husky-based auto car model loaded into the PyBullet simulation.

- **Camera and Visualization:**  
  - An onboard camera provides a live view from the car’s perspective.
  - In simulator mode, the camera image is displayed using OpenCV.

- **Discrete Action Mapping:**  
  The control actions are defined as:
  - **0:** No Action (maintain current velocity)
  - **1:** Brake (reduce velocity rapidly)
  - **2:** Increase velocity (smooth acceleration)
  - **3:** Decrease velocity (smooth deceleration)
  - **4:** Turn left (gentle turn)
  - **5:** Turn right (gentle turn)
  - **6:** In-place turn (left)
  - **7:** In-place turn (right)

- **Reward Function:**  
  The simulation divides the room into a 100×100 grid. Rewards are assigned based on exploration:
  - New cell: **+100**
  - Same cell as previous: **–5**
  - Previously visited but different cell: **–1**

### 2. Physical Raspberry Pi Car

- **Hardware Components:**  
  - Raspberry Pi (e.g., Raspberry Pi 3 or 4)
  - Motor driver board (e.g., L298N or equivalent)
  - Camera module (e.g., Web Camera)

- **Integration:**  
  - The physical car uses control algorithms similar to the simulation, enabling a smooth transition from virtual to real-world testing.
  - Image data is used to adjust the control commands, ensuring robust navigation in a real environment.

- **Deployment:**  
  - After developing and testing policies in simulation, these can be ported to the Raspberry Pi.
  - This allows for real-world validation and fine-tuning of the exploration strategy.

## Installation and Setup

### Prerequisites

- **Python 3.11+**