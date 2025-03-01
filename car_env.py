import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import gymnasium as gym  # Correct import for Gymnasium
from gymnasium import spaces # Correct import for Gymnasium spaces

class AutoCarBase:
    """
    Base class that sets up the simulation world and handles wheel/velocity management.
    """
    def __init__(self, connection_mode=p.GUI, step_time=1/60.):
        self.connection_mode = connection_mode
        p.connect(self.connection_mode)
        p.setPhysicsEngineParameter(fixedTimeStep=step_time)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        # For RL, use manual stepping
        p.setRealTimeSimulation(0)

        # Camera parameters
        self.camera_height = 0.3
        self.current_fov = 60

        # Velocity management parameters
        self.current_velocity = 0.0
        self.acceleration_rate = 0.5    # Smooth acceleration increment
        self.max_forward_velocity = 10.0
        self.max_backward_velocity = -10.0
        self.brake_rate = self.acceleration_rate * 3.0
        self.maxForce = 100  # Maximum force for wheel motors
        self.room_size = 20  # Room size (assumed square)

        # Set up the simulation environment and load the car model
        self._setup_environment()
        self.car = self._load_car()

    def _setup_environment(self):
        """Set up ground, walls, and obstacles."""
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

    def _load_car(self, seed=None):
        """Load the Husky car model."""
        carpos = [0, 0, 0.01]
        if seed is not None:
            np.random.seed(seed)
            carpos[0] = np.random.uniform(-self.room_size/2, self.room_size/2)
            carpos[1] = np.random.uniform(-self.room_size/2, self.room_size/2)
        car = p.loadURDF("urdf/husky/husky.urdf", carpos[0], carpos[1], carpos[2])
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

    def disconnect(self):
        p.disconnect()

    def reset(self, seed=None):
        """Reset the simulation environment."""
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self._setup_environment()
        self.car = self._load_car(seed)
        self.current_velocity = 0.0
        return self.get_camera_image()

    def set_wheel_velocity(self, left_multiplier=1.0, right_multiplier=1.0):
        """Set the wheel velocities based on the current velocity and multipliers."""
        for joint in [2, 4]:
            p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL,
                                    targetVelocity=self.current_velocity * left_multiplier, force=self.maxForce)
        for joint in [3, 5]:
            p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL,
                                    targetVelocity=self.current_velocity * right_multiplier, force=self.maxForce)

    def _increase_velocity(self):
        self.current_velocity += self.acceleration_rate
        self.current_velocity = min(self.max_forward_velocity, self.current_velocity)

    def _decrease_velocity(self):
        self.current_velocity -= self.acceleration_rate
        self.current_velocity = max(self.max_backward_velocity, self.current_velocity)

    def _brake(self):
        if self.current_velocity > 0:
            self.current_velocity = max(0, self.current_velocity - self.brake_rate)
        elif self.current_velocity < 0:
            self.current_velocity = min(0, self.current_velocity + self.brake_rate)

    def apply_action(self, action):
        """Apply a discrete action to the car."""
        if action == 0:
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
            self.set_wheel_velocity(left_multiplier=0.2, right_multiplier=1.0)
        elif action == 5:
            self.set_wheel_velocity(left_multiplier=1.0, right_multiplier=0.2)
        elif action == 6:
            self.set_wheel_velocity(left_multiplier=-1.0, right_multiplier=1.0)
        elif action == 7:
            self.set_wheel_velocity(left_multiplier=1.0, right_multiplier=-1.0)
        else:
            self.set_wheel_velocity(0, 0)

class AutoCarSimulator(AutoCarBase):
    """
    Simulator tool that processes keyboard inputs for real-time control.
    """
    def __init__(self):
        super().__init__(connection_mode=p.GUI, step_time=1/60.)
        p.setRealTimeSimulation(1)

    def process_keyboard(self):
        keys = p.getKeyboardEvents()
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

        action = None
        if 48 in keys and (keys[48] & p.KEY_WAS_TRIGGERED):
            action = 0
        elif 49 in keys and (keys[49] & p.KEY_WAS_TRIGGERED):
            action = 1
        elif 50 in keys and (keys[50] & p.KEY_WAS_TRIGGERED):
            action = 2
        elif 51 in keys and (keys[51] & p.KEY_WAS_TRIGGERED):
            action = 3
        elif 52 in keys and (keys[52] & p.KEY_WAS_TRIGGERED):
            action = 4
        elif 53 in keys and (keys[53] & p.KEY_WAS_TRIGGERED):
            action = 5
        elif 54 in keys and (keys[54] & p.KEY_WAS_TRIGGERED):
            action = 6
        elif 55 in keys and (keys[55] & p.KEY_WAS_TRIGGERED):
            action = 7

        if action is not None:
            self.apply_action(action)
        if 27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED):
            return False
        return True

    def run(self):
        while True:
            if not self.process_keyboard():
                break
            p.stepSimulation()
            _ = self.get_camera_image()  # Display camera image if needed
        self.disconnect()

class AutoCarEnv(AutoCarBase, gym.Env): # CORRECTED INHERITANCE ORDER: AutoCarBase, gym.Env - AutoCarBase init runs first
    """
    Gym-style RL environment using discrete actions.
    Observations are RGB images from the car's camera.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False, image_size=(84, 84)):
        mode = p.GUI if render else p.DIRECT
        AutoCarBase.__init__(self, connection_mode=mode, step_time=1/60.) # Call AutoCarBase __init__ FIRST
        gym.Env.__init__(self) # Then call gym.Env __init__
        self.image_width, self.image_height = image_size
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.float32
        )
        self._max_steps = 10000
        self.current_step = 0
        self.visited = set()
        self.last_cell = None
        self.grid_size = self.room_size * 10  # e.g. 200x200 grid
        self.reward_new_cell = 100
        self.reward_same_cell = -5
        self.reward_revisit_cell = -1
        self.seed_value = None

    def _get_cell_from_pos(self, pos):
        grid_x = int((pos[0] + self.room_size/2) / self.room_size * self.grid_size)
        grid_y = int((pos[1] + self.room_size/2) / self.room_size * self.grid_size)
        grid_x = max(0, min(self.grid_size-1, grid_x))
        grid_y = max(0, min(self.grid_size-1, grid_y))
        return (grid_x, grid_y)

    def seed(self, seed=None):
        self.seed_value = seed

    def reset(self, seed=None, options=None): # CORRECTED reset signature in AutoCarEnv
        gym.Env.reset(self, seed=seed, options=options) # Call superclass reset (Gym.Env) - this is important for seed handling
        obs = AutoCarBase.reset(self, seed=seed)
        self.current_step = 0
        self.visited = set()
        self.last_cell = None
        pos, _ = p.getBasePositionAndOrientation(self.car)
        cell = self._get_cell_from_pos(pos)
        self.visited.add(cell)
        self.last_cell = cell
        # Capture observation with reduced resolution
        obs = self.get_camera_image(width=self.image_width, height=self.image_height)
        return obs, {} # Correct return: observation, info

    def step(self, action):
        self.apply_action(action)
        p.stepSimulation()
        obs = self.get_camera_image(width=self.image_width, height=self.image_height)
        pos, _ = p.getBasePositionAndOrientation(self.car)
        cell = self._get_cell_from_pos(pos)
        if cell not in self.visited:
            reward = self.reward_new_cell
            self.visited.add(cell)
        elif cell == self.last_cell:
            reward = self.reward_same_cell
        else:
            reward = self.reward_revisit_cell
        self.last_cell = cell
        self.current_step += 1
        terminated = self.current_step >= self._max_steps # Corrected variable name to 'terminated'
        truncated = False # Add truncated if needed, if episode can end for other reasons than reaching max steps
        return obs, reward, terminated, truncated, {} # Correct return: observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        return self.get_camera_image(width=self.image_width, height=self.image_height)

    def close(self):
        self.disconnect()

if __name__ == "__main__":
    # Run the simulator with keyboard controls
    sim = AutoCarSimulator()
    sim.reset(seed=100)
    sim.run()
    # For gym-style random actions, uncomment below:
    # env = AutoCarEnv(render=True, image_size=(256, 256))
    # obs = env.reset()
    # sum = 0
    # for _ in range(1000):
    #     action = 2 # env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)
    #     if done:
    #         obs = env.reset()
    #     sum += reward
    #     print(sum)
    # env.close()