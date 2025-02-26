import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import gym
from gym import spaces

class AutoCarBase:
    """
    Base class that sets up the simulation world and handles all wheel/velocity management.
    It maintains a current velocity (with smooth acceleration/deceleration) and exposes helper
    functions to apply actions.
    """
    def __init__(self, connection_mode=p.GUI):
        self.connection_mode = connection_mode
        p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        # For RL, we use manual stepping
        p.setRealTimeSimulation(0)

        # Camera parameters
        self.camera_height = 0.3
        self.current_fov = 60

        # Velocity management parameters
        self.current_velocity = 0.0
        self.acceleration_rate = 0.5    # smooth acceleration increment
        self.max_forward_velocity = 10.0
        self.max_backward_velocity = -10.0

        # Brake rate (for sudden deceleration)
        self.brake_rate = self.acceleration_rate * 3.0

        self.maxForce = 100  # Maximum force for wheel motors
        self.room_size = 20  # Room size (assumed square)

        # Set up the simulation environment and load the car model
        self._setup_environment()
        self.car = self._load_car()

    def _setup_environment(self):
        """Set up ground, walls, and obstacles."""
        # Ground plane with texture
        self.planeId = p.loadURDF("plane.urdf")
        floor_texture = p.loadTexture("urdf/floor.jpeg")
        p.changeVisualShape(self.planeId, -1, textureUniqueId=floor_texture)

        # Walls around the room
        wall_thickness = 0.2
        wall_height = 5
        wall_shapes = [
            [self.room_size, wall_thickness, wall_height],
            [self.room_size, wall_thickness, wall_height],
            [wall_thickness, self.room_size, wall_height],
            [wall_thickness, self.room_size, wall_height]
        ]
        wall_positions = [
            [0, self.room_size / 2, wall_height / 2],
            [0, -self.room_size / 2, wall_height / 2],
            [self.room_size / 2, 0, wall_height / 2],
            [-self.room_size / 2, 0, wall_height / 2]
        ]
        wall_texture = p.loadTexture("urdf/wall.jpeg")
        for pos, size in zip(wall_positions, wall_shapes):
            wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            wall_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=wall_collision,
                                        basePosition=pos)
            p.changeVisualShape(wall_id, -1, textureUniqueId=wall_texture)

        # Obstacles: a desk and a chair
        desk_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5, 0.7, 0.4])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=desk_collision, basePosition=[3, 3, 0.4])

        chair_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.8])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=chair_collision, basePosition=[-2, -3, 0.8])

    def _load_car(self):
        """Load the Husky car model."""
        carpos = [0, 0, 0.1]
        car = p.loadURDF("husky/husky.urdf", carpos[0], carpos[1], carpos[2])
        return car

    def get_camera_params(self):
        """Compute camera position and target based on the car's pose."""
        pos, orn_quat = p.getBasePositionAndOrientation(self.car)
        orn = list(p.getEulerFromQuaternion(orn_quat))
        front_cam = [0.345 * math.cos(orn[2]),
                     0.345 * math.sin(orn[2]),
                     self.camera_height]
        camera_pos = [pos[i] + front_cam[i] for i in range(3)]
        camera_target = [pos[0] + math.cos(orn[2]),
                         pos[1] + math.sin(orn[2]),
                         self.camera_height]
        return camera_pos, camera_target, self.current_fov

    def get_camera_image(self, width=256, height=256):
        """Capture an RGB camera image from the car's perspective."""
        camera_pos, camera_target, fov = self.get_camera_params()
        aspect = width / height
        near = 0.1
        far = 20
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos,
                                          cameraTargetPosition=camera_target,
                                          cameraUpVector=[0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov=fov,
                                                         aspect=aspect,
                                                         nearVal=near,
                                                         farVal=far)
        images = p.getCameraImage(width, height, view_matrix, projection_matrix,
                                  shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.reshape(images[2], (height, width, 4))[:, :, :3] / 255.0
        return rgb_array

    def step_simulation(self, sleep_time=1./240.):
        """Advance the simulation one step."""
        p.stepSimulation()
        time.sleep(sleep_time)

    def disconnect(self):
        p.disconnect()

    def reset(self):
        """Reset the simulation environment."""
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self._setup_environment()
        self.car = self._load_car()
        self.current_velocity = 0.0
        return self.get_camera_image()

    def set_wheel_velocity(self, left_multiplier=1.0, right_multiplier=1.0):
        """
        Set the wheel velocities based on the current velocity and provided multipliers.
        Wheels with indices 2 and 4 are on one side; 3 and 5 on the other.
        """
        for joint in [2, 4]:
            p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL,
                                    targetVelocity=self.current_velocity * left_multiplier, force=self.maxForce)
        for joint in [3, 5]:
            p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL,
                                    targetVelocity=self.current_velocity * right_multiplier, force=self.maxForce)
    
    def _increase_velocity(self):
        """Smoothly increase the current velocity. If currently moving backward, brake first."""
        self.current_velocity += self.acceleration_rate
        self.current_velocity = min(self.max_forward_velocity, self.current_velocity)

    def _decrease_velocity(self):
        """Smoothly decrease the current velocity. If moving forward, reduce until zero."""
        self.current_velocity -= self.acceleration_rate
        self.current_velocity = max(self.max_backward_velocity, self.current_velocity)

    def _brake(self):
        """Apply a strong deceleration to quickly reduce velocity toward zero."""
        if self.current_velocity > 0:
            self.current_velocity = max(0, self.current_velocity - self.brake_rate)
        elif self.current_velocity < 0:
            self.current_velocity = min(0, self.current_velocity + self.brake_rate)

    def apply_action(self, action):
        """
        Apply a discrete action to the car.
        New Action mapping:
          0: No Action (maintain current velocity)
          1: Brake (reduce velocity sharpely)
          2: Increase velocity (smoothly)
          3: Decrease velocity (smoothly)
          4: Turn left
          5: Turn right
          6: In-place turn (left)
          7: In-place turn (right)
        """
        if action == 0:
            # No Action: simply maintain the current wheel velocity.
            self.set_wheel_velocity()
        elif action == 1:
            self._brake()
            self.set_wheel_velocity()
        elif action == 2:
            self._increase_velocity()
            self.set_wheel_velocity()
        elif action == 3:
            self._decrease_velocity()
            self.set_wheel_velocity()
        elif action == 4:
            # Turn left: adjust multipliers for a gentle turn.
            self.set_wheel_velocity(left_multiplier=0.2, right_multiplier=1.0)
        elif action == 5:
            # Turn right: adjust multipliers for a gentle turn.
            self.set_wheel_velocity(left_multiplier=1.0, right_multiplier=0.2)
        elif action == 6:
            # In-place turn left: left wheels reverse, right wheels forward.
            self.set_wheel_velocity(left_multiplier=-1.0, right_multiplier=1.0)
        elif action == 7:
            # In-place turn right: left wheels forward, right wheels reverse.
            self.set_wheel_velocity(left_multiplier=1.0, right_multiplier=-1.0)
        else:
            # Invalid action: stop the wheels.
            self.set_wheel_velocity(0, 0)

class AutoCarSimulator(AutoCarBase):
    """
    Simulator tool that processes keyboard events and maps them to discrete actions.
    The mapping here uses the number keys 0-7 as defined:
      0: No Action
      1: Brake (reduce velocity largely)
      2: Increase velocity (smoothly)
      3: Decrease velocity (smoothly)
      4: Turn left
      5: Turn right
      6: In-place turn (left)
      7: In-place turn (right)
    Additionally, global camera adjustments remain available:
      Z/X: decrease/increase FOV
      Q/E: lower/raise camera height
    """
    def __init__(self):
        super().__init__(connection_mode=p.GUI)
        p.setRealTimeSimulation(1)

    def process_keyboard(self):
        keys = p.getKeyboardEvents()
        # Global camera adjustments
        if 122 in keys and (keys[122] & p.KEY_IS_DOWN):  # 'Z'
            self.current_fov = max(30, self.current_fov - 5)
            print("FOV decreased to:", self.current_fov)
        if 120 in keys and (keys[120] & p.KEY_IS_DOWN):  # 'X'
            self.current_fov = min(120, self.current_fov + 5)
            print("FOV increased to:", self.current_fov)
        if 113 in keys and (keys[113] & p.KEY_IS_DOWN):  # 'Q'
            self.camera_height = max(0.1, self.camera_height - 0.05)
            print("Camera height lowered to:", self.camera_height)
        if 101 in keys and (keys[101] & p.KEY_IS_DOWN):  # 'E'
            self.camera_height = min(1.0, self.camera_height + 0.05)
            print("Camera height raised to:", self.camera_height)

        # Map numeric keys (ASCII codes 48-'0' to 55-'7') to actions.
        action = None
        if 48 in keys and (keys[48] & p.KEY_WAS_TRIGGERED):  # '0'
            action = 0
        elif 49 in keys and (keys[49] & p.KEY_WAS_TRIGGERED):  # '1'
            action = 1
        elif 50 in keys and (keys[50] & p.KEY_WAS_TRIGGERED):  # '2'
            action = 2
        elif 51 in keys and (keys[51] & p.KEY_WAS_TRIGGERED):  # '3'
            action = 3
        elif 52 in keys and (keys[52] & p.KEY_WAS_TRIGGERED):  # '4'
            action = 4
        elif 53 in keys and (keys[53] & p.KEY_WAS_TRIGGERED):  # '5'
            action = 5
        elif 54 in keys and (keys[54] & p.KEY_WAS_TRIGGERED):  # '6'
            action = 6
        elif 55 in keys and (keys[55] & p.KEY_WAS_TRIGGERED):  # '7'
            action = 7

        if action is not None:
            self.apply_action(action)

        # Exit simulation if ESC is pressed
        if 27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED):
            return False
        return True

    def run(self):
        """Run the simulation loop until exit, displaying the camera image."""
        while True:
            if not self.process_keyboard():
                break
            self.step_simulation()
            # Get and display the camera image
            _ = self.get_camera_image()
           
        self.disconnect()

class AutoCarEnv(gym.Env, AutoCarBase):
    """
    Gym-style RL environment using the discrete action mapping:
      0: No Action
      1: Brake (reduce velocity largely)
      2: Increase velocity (smoothly)
      3: Decrease velocity (smoothly)
      4: Turn left
      5: Turn right
      6: In-place turn (left)
      7: In-place turn (right)
    The goal is to visit all areas of the room. The room (assumed to be from -10 to +10 in x and y)
    is divided into a 200x200 grid. The reward is assigned as follows:
      - If the car visits a new grid cell: +100
      - If the car remains in the same cell as the previous step: -5
      - If the car enters a cell that was already visited (but is different from the last cell): -1
    Observations are RGB images from the car's camera.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False):
        mode = p.GUI if render else p.DIRECT
        AutoCarBase.__init__(self, connection_mode=mode)
        gym.Env.__init__(self)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(256, 256, 3), dtype=np.float32)
        self._max_steps = 10000
        self.current_step = 0
        # Tracking visited grid cells
        self.visited = set()
        self.last_cell = None
        # Grid size for room division
        self.grid_size = self.room_size * 10 # 200 x 200
        # Reward for visiting a new cell
        self.reward_new_cell = 100
        # Reward for staying in the same cell
        self.reward_same_cell = -5
        # Reward for revisiting a cell
        self.reward_revisit_cell = -1
        
    def _get_cell_from_pos(self, pos):
        """
        Convert the car's (x, y) position into a grid cell index.
        Assumes the room spans from -room_size/2 to room_size/2 in both x and y.
        """
        grid_x = int((pos[0] + self.room_size/2) / self.room_size * self.grid_size)
        grid_y = int((pos[1] + self.room_size/2) / self.room_size * self.grid_size)
        # Ensure indices are within bounds [0, grid_size-1]
        grid_x = max(0, min(self.grid_size-1, grid_x))
        grid_y = max(0, min(self.grid_size-1, grid_y))
        return (grid_x, grid_y)

    def reset(self):
        obs = AutoCarBase.reset(self)
        self.current_step = 0
        self.visited = set()
        self.last_cell = None
        # Initialize visited cell based on starting position
        pos, _ = p.getBasePositionAndOrientation(self.car)
        cell = self._get_cell_from_pos(pos)
        self.visited.add(cell)
        self.last_cell = cell
        return obs

    def step(self, action):
        # Apply the action using centralized wheel management
        self.apply_action(action)
        self.step_simulation()
        obs = self.get_camera_image()

        # Get the current cell from car's position
        pos, _ = p.getBasePositionAndOrientation(self.car)
        cell = self._get_cell_from_pos(pos)

        # Determine reward based on grid cell visitation
        if cell not in self.visited:
            reward = self.reward_new_cell
            self.visited.add(cell)
        elif cell == self.last_cell:
            reward = self.reward_same_cell
        else:
            reward = self.reward_revisit_cell

        self.last_cell = cell

        self.current_step += 1
        done = self.current_step >= self._max_steps
        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.get_camera_image()
        elif mode == 'human':
            return self.get_camera_image()

    def close(self):
        self.disconnect()


if __name__ == "__main__":
    # Run the simulator with keyboard controls
    sim = AutoCarSimulator()
    sim.run()

    # Run the environment with Gym-style actions
    # env = AutoCarEnv(render=True)
    # obs = env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)
    #     if done:
    #         obs = env.reset()
    # env.close()