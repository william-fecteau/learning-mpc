from matplotlib import pyplot as plt
import numpy as np

def find_closest_waypoint_in_path(path, point):
    """
    Finds the closest waypoint in a given path to a specified point.

    Args:
    - path (np.ndarray): An array of shape (N, D), where N is the number of waypoints
                         and D is the dimension (2D or 3D).
    - point (np.ndarray): A point of shape (D,) to compare against the path.

    Returns:
    - tuple: (closest_point (np.ndarray), index (int)) where closest_point is the 
             nearest waypoint in the path and index is its position.
    """
    distances = np.linalg.norm(path - point, axis=1)
    i = np.argmin(distances)
    return path[i], i

def project_point_onto_line_segment(p, a, b):
    """
    Projects a point p onto the line segment defined by points a and b.

    Args:
    - p (np.ndarray): The point to project.
    - a (np.ndarray): The start point of the line segment.
    - b (np.ndarray): The end point of the line segment.

    Returns:
    - np.ndarray: The projected point on the line segment.
    """
    ap = p - a
    ab = b - a
    ab_norm = np.dot(ab, ab)
    if ab_norm == 0:
        return a  # a and b are the same point

    t = np.dot(ap, ab) / ab_norm
    t = np.clip(t, 0, 1)
    projection = a + t * ab
    return projection

def cartesian_to_frenet(path, point):
    """
    Converts a Cartesian point to a Frenet coordinate relative to a given path.

    Args:
    - path (np.ndarray): An array of waypoints (shape: [N, D]) representing the path. D is the dimension (2D or 3D)
    - point_to_convert (np.ndarray): The Cartesian point to convert (shape: [D]). D is the dimension (2D or 3D)

    Returns:
    - np.ndarray: Frenet coordinates [s, d], where:
        - s is the index of the closest waypoint in the path,
        - d is the perpendicular distance from the path segment to the point.
    """
    closest_point, s = find_closest_waypoint_in_path(path, point)
    next_point = path[(s+1)%path.shape[0]]
    previous_point = path[(s-1)%path.shape[0]]

    point_on_segment = project_point_onto_line_segment(closest_point, previous_point, next_point)
    d = np.linalg.norm(point - point_on_segment)

    return np.array([s, d])


if __name__ == '__main__':
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    path = np.column_stack((x, y))

    point_to_convert = np.array([0.3, 0.5])

    frenet_point = cartesian_to_frenet(path, point_to_convert)
    print(frenet_point)

    s_point = path[int(frenet_point[0])]
    d = frenet_point[1]

    print(f's={frenet_point[0]},d={frenet_point[1]}')
    print(f'Sanity check:{np.linalg.norm(s_point - point_to_convert)}')

    plt.scatter(path[:, 0], path[:, 1], label='Path')
    plt.scatter(*point_to_convert, color="red", label='Point to convert')
    plt.scatter(*s_point, color='green', label='s point')
    plt.legend()
    plt.show()