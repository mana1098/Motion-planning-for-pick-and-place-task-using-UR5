import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, q):
        self.q = q  # Configuration
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, joint_limits, max_iterations=1000, step_size=0.1):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacle_list = obstacle_list
        self.joint_limits = joint_limits
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.node_list = [self.start]

    def plan(self):
        for _ in range(self.max_iterations):
            q_rand = self.random_config()
            nearest_node = self.nearest_neighbor(q_rand)
            q_new = self.new_config(nearest_node.q, q_rand)

            if self.is_collision_free(q_new):
                new_node = Node(q_new)
                new_node.parent = nearest_node
                self.node_list.append(new_node)

                if self.distance(new_node.q, self.goal.q) < self.step_size:
                    self.goal.parent = new_node
                    return self.extract_path()

        return None  # Failed to find a path

    def random_config(self):
        return np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def nearest_neighbor(self, q):
        distances = [self.distance(node.q, q) for node in self.node_list]
        return self.node_list[np.argmin(distances)]

    def new_config(self, q_near, q_rand):
        q_new = q_near + self.step_size * (q_rand - q_near) / self.distance(q_near, q_rand)
        return np.clip(q_new, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def is_collision_free(self, q):
        # Simplified collision checking
        for obstacle in self.obstacle_list:
            if self.distance(q, obstacle) < 0.1:  # Adjust threshold as needed
                return False
        return True

    def distance(self, q1, q2):
        return np.linalg.norm(q1 - q2)

    def extract_path(self):
        path = [self.goal.q]
        node = self.goal
        while node.parent is not None:
            node = node.parent
            path.append(node.q)
        path.reverse()
        return np.array(path)

class UR5Kinematics:
    def __init__(self):
        # UR5 DH parameters
        self.d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self.a = [0, -0.425, -0.39225, 0, 0, 0]
        self.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i in range(6):
            T = np.dot(T, self.dh_matrix(q[i], self.d[i], self.a[i], self.alpha[i]))
        return T[:3, 3]  # Return only the position

    def dh_matrix(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

def pick_and_place(start, pick, place, obstacle_list):
    ur5 = UR5Kinematics()
    joint_limits = np.array([[-np.pi, np.pi]] * 6)  # Simplified joint limits

    # Plan path from start to pick position
    rrt_to_pick = RRT(start, pick, obstacle_list, joint_limits)
    path_to_pick = rrt_to_pick.plan()

    # Plan path from pick to place position
    rrt_to_place = RRT(pick, place, obstacle_list, joint_limits)
    path_to_place = rrt_to_place.plan()

    if path_to_pick is None or path_to_place is None:
        print("Failed to find a path")
        return None

    # Combine paths
    full_path = np.vstack((path_to_pick, path_to_place))

    # Convert joint space path to Cartesian space
    cartesian_path = np.array([ur5.forward_kinematics(q) for q in full_path])

    return cartesian_path

# Example usage
start = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
pick = np.array([np.pi/4, -np.pi/3, np.pi/3, -np.pi/2, -np.pi/4, 0])
place = np.array([-np.pi/4, -np.pi/4, np.pi/4, -np.pi/3, np.pi/4, 0])
obstacle_list = [np.array([0.2, 0.2, 0.2]), np.array([-0.2, 0.3, 0.4])]  # Example obstacles

cartesian_path = pick_and_place(start, pick, place, obstacle_list)

if cartesian_path is not None:
    # Visualize the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cartesian_path[:, 0], cartesian_path[:, 1], cartesian_path[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('UR5 Pick and Place Path')
    plt.show()
