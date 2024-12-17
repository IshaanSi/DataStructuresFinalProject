import laspy
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial import KDTree


class Vertex:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.edges = []

    def add_edge(self, vertex):
        self.edges.append(vertex)

    def __repr__(self):
        return f"Vertex(id={self.id}, x={self.x}, y={self.y}, z={self.z}, edges={[v.id for v in self.edges]})"


class Edge:
    def __init__(self, fro, to):
        self.fro = fro
        self.to = to
        self.weight = math.sqrt((fro.x - to.x)**2 + (fro.y - to.y)**2 + (fro.z - to.z)**2)

    def __repr__(self):
        return f"Edge(from={self.fro.id}, to={self.to.id}, weight={self.weight:.2f})"

class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.weights = []



def plot_surface_with_edges(vertices, path1=None, path2=None, path3=None):
    plt.style.use('rose-pine-matplotlib-main/themes/rose-pine-moon.mplstyle')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = np.array([v.x for v in vertices])
    y_vals = np.array([v.y for v in vertices])
    z_vals = np.array([v.z for v in vertices])

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_vals.min(), x_vals.max(), 50),
        np.linspace(y_vals.min(), y_vals.max(), 50)
    )

    z_grid = griddata(
        points=(x_vals, y_vals),
        values=z_vals,
        xi=(x_grid, y_grid),
        method='linear'
    )

    surface = ax.plot_surface(
        x_grid, y_grid, z_grid,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9
    )
    for v in vertices:
        for neighbor in v.edges:
            ax.plot(
                [v.x, neighbor.x],
                [v.y, neighbor.y],
                [v.z, neighbor.z],
                color='gray',
                linewidth=0.5,
                alpha=0.8
            )
    if path1:
        path1_vertices = [next(v for v in vertices if v.id == p_id) for p_id in path1]
        path1_x = [v.x for v in path1_vertices]
        path1_y = [v.y for v in path1_vertices]
        path1_z = [v.z for v in path1_vertices]

        ax.plot(
            path1_x, path1_y, path1_z,
            color='#f4a261', linewidth=3, label='Path 1'
        )
    if path2:
        path2_vertices = [next(v for v in vertices if v.id == p_id) for p_id in path2]
        path2_x = [v.x for v in path2_vertices]
        path2_y = [v.y for v in path2_vertices]
        path2_z = [v.z for v in path2_vertices]

        ax.plot(
            path2_x, path2_y, path2_z,
            color='#2a9d8f', linewidth=3, label='Path 2'
        )
    if path3:
        path3_vertices = [next(v for v in vertices if v.id == p_id) for p_id in path3]
        path3_x = [v.x for v in path3_vertices]
        path3_y = [v.y for v in path3_vertices]
        path3_z = [v.z for v in path3_vertices]

        ax.plot(
            path3_x, path3_y, path3_z,
            color='#264653', linewidth=3, label='Path 3'
        )
    colorbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    colorbar.set_label('Z Value', color='white', fontsize=12)

    ax.set_xlabel('X', fontsize=12, color='white')
    ax.set_ylabel('Y', fontsize=12, color='white')
    ax.set_zlabel('Z', fontsize=12, color='white')
    ax.set_facecolor('#191724')
    ax.grid(False)
    ax.tick_params(colors='white', labelsize=10)
    if path1 or path2 or path3:
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.show()


def load_data(sample_size, random=False):
    if random == False:
        print("Gathering a grid sample")
        with laspy.open("Umbagog LiDAR 2016.laz") as lidar_file:
            point_cloud = lidar_file.read()
        x, y, z = point_cloud.x, point_cloud.y, point_cloud.z

        grid_size = int(np.sqrt(sample_size))
        if grid_size ** 2 < sample_size:
            grid_size += 1 

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        grid_points = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
        lidar_points = np.vstack([x, y]).T
        tree = KDTree(lidar_points)

        _, indices = tree.query(grid_points)

        z_sample = z[indices]
        return grid_points[:, 0], grid_points[:, 1], z_sample

    else:
        with laspy.open("Umbagog LiDAR 2016.laz") as lidar_file:
            point_cloud = lidar_file.read()
        x, y, z = point_cloud.x, point_cloud.y, point_cloud.z 

        indices = np.random.choice(len(x), size=sample_size, replace=False)
        x_sample, y_sample, z_sample = x[indices], y[indices], z[indices] ######################## random
        return x_sample, y_sample, z_sample


def create_vertices(x_sample, y_sample, z_sample):
    vertices = []
    for i, (x, y, z) in enumerate(zip(x_sample, y_sample, z_sample)):
        vertices.append(Vertex(id=i, x=x, y=y, z=z))
    return vertices


# attempting to find the 4 closest neighbords (front, back, sides)
def find_closest_neighbors(vertices):
    for vertex in tqdm(vertices, desc="Processing vertices", unit="vertex"):
        distances = []
        for other in vertices:
            if vertex.id != other.id:
                dist = np.sqrt((vertex.x - other.x)**2 + (vertex.y - other.y)**2 + (vertex.z - other.z)**2)
                distances.append((dist, other))
        distances.sort(key=lambda x: x[0])
        closest_neighbors = [d[1] for d in distances[:4]]
        for neighbor in closest_neighbors:
            vertex.add_edge(neighbor)

# finds neighbors in a grid
def find_grid_neighbors(vertices):
    for vertex in tqdm(vertices, desc="Processing vertices", unit="vertex"):
        top_neighbor = None
        bottom_neighbor = None
        left_neighbor = None
        right_neighbor = None


        min_top_dist = float("inf")
        min_bottom_dist = float("inf")
        min_left_dist = float("inf")
        min_right_dist = float("inf")

        for other in vertices:
            if vertex.id != other.id:
                dx = other.x - vertex.x     # horizontal distance of the vertex
                dy = other.y - vertex.y


                if dx == 0 and dy > 0 and dy < min_top_dist:  # front
                    min_top_dist = dy
                    top_neighbor = other
                elif dx == 0 and dy < 0 and abs(dy) < min_bottom_dist:  # back
                    min_bottom_dist = abs(dy)
                    bottom_neighbor = other
                elif dy == 0 and dx < 0 and abs(dx) < min_left_dist:  # left
                    min_left_dist = abs(dx)
                    left_neighbor = other
                elif dy == 0 and dx > 0 and dx < min_right_dist:  # right
                    min_right_dist = dx
                    right_neighbor = other
        for neighbor in [top_neighbor, bottom_neighbor, left_neighbor, right_neighbor]:
            if neighbor is not None:
                vertex.add_edge(neighbor)


def create_graph(vertices):
    graph = Graph()
    graph.vertices = vertices
    print("Creating edges:")
    for vertex in tqdm(vertices, desc="Adding edges", unit="vertex"):
        for neighbor in vertex.edges:
            edge = Edge(fro=vertex, to=neighbor)
            graph.edges.append(edge)
            graph.weights.append(edge.weight)
    return graph

def find_shortest_path(graph, start_id, end_id):
    distances = {vertex.id: float('inf') for vertex in graph.vertices}
    previous = {vertex.id: None for vertex in graph.vertices}
    distances[start_id] = 0
    pq = [(0, start_id)] 

    while pq:
        current_distance, current_id = heapq.heappop(pq)
        if current_distance > distances[current_id]:
            continue

        current_vertex = next(v for v in graph.vertices if v.id == current_id)

        for neighbor in current_vertex.edges:
            edge = next(e for e in graph.edges if e.fro == current_vertex and e.to == neighbor)
            new_distance = current_distance + edge.weight

            if new_distance < distances[neighbor.id]:
                distances[neighbor.id] = new_distance
                previous[neighbor.id] = current_id
                heapq.heappush(pq, (new_distance, neighbor.id))

        if current_id == end_id:
            break

# reconstructing path
    path = []
    current = end_id
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distances[end_id]

def find_shortest_path_vertical(graph, start_id, end_id):
    distances = {vertex.id: float('inf') for vertex in graph.vertices}
    previous = {vertex.id: None for vertex in graph.vertices}
    distances[start_id] = 0
    pq = [(0, start_id)]

    while pq:
        current_distance, current_id = heapq.heappop(pq)
        if current_distance > distances[current_id]:
            continue

        current_vertex = next(v for v in graph.vertices if v.id == current_id)
        for neighbor in current_vertex.edges:
            edge = next(e for e in graph.edges if e.fro == current_vertex and e.to == neighbor)
            vertical_distance = abs(current_vertex.z - neighbor.z)  # minimizing vertical distance
            new_distance = current_distance + vertical_distance

            if new_distance < distances[neighbor.id]:
                distances[neighbor.id] = new_distance
                previous[neighbor.id] = current_id
                heapq.heappush(pq, (new_distance, neighbor.id))

        if current_id == end_id:
            break


    path = []
    current = end_id
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distances[end_id]

def find_shortest_path_horizontal(graph, start_id, end_id):
    distances = {vertex.id: float('inf') for vertex in graph.vertices}
    previous = {vertex.id: None for vertex in graph.vertices}
    distances[start_id] = 0
    pq = [(0, start_id)] 

    while pq:
        current_distance, current_id = heapq.heappop(pq)


        if current_distance > distances[current_id]:
            continue


        current_vertex = next(v for v in graph.vertices if v.id == current_id)

        for neighbor in current_vertex.edges:
            edge = next(e for e in graph.edges if e.fro == current_vertex and e.to == neighbor)
            horizontal_distance = math.sqrt((current_vertex.x - neighbor.x) ** 2 + (current_vertex.y - neighbor.y) ** 2) # horizontal distance this time
            new_distance = current_distance + horizontal_distance

            if new_distance < distances[neighbor.id]:
                distances[neighbor.id] = new_distance
                previous[neighbor.id] = current_id
                heapq.heappush(pq, (new_distance, neighbor.id))

        if current_id == end_id:
            break

    path = []
    current = end_id
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distances[end_id]

def find_longest_path(graph, start_id, end_id):
    distances = {vertex.id: float('-inf') for vertex in graph.vertices}
    previous = {vertex.id: None for vertex in graph.vertices}
    distances[start_id] = 0
    pq = [(0, start_id)] # distance, id
    
    visited = set()

    while pq:
        current_distance, current_id = heapq.heappop(pq)
        if current_id in visited:
            continue
        
        visited.add(current_id)

        current_vertex = next(v for v in graph.vertices if v.id == current_id)

        for neighbor in current_vertex.edges:
            if neighbor.id in visited:
                continue
            
            edge = next(e for e in graph.edges if e.fro == current_vertex and e.to == neighbor)
            new_distance = current_distance + edge.weight

            if new_distance > distances[neighbor.id]:  # maximizing distance
                distances[neighbor.id] = new_distance
                previous[neighbor.id] = current_id
                heapq.heappush(pq, (new_distance, neighbor.id))

        if current_id == end_id:
            break

    # Reconstruct the path
    path = []
    current = end_id
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distances[end_id]


# perfect square ideally, less data loss
sample_size = 1600
x_sample,y_sample,z_sample = load_data(sample_size, random=False)
print(type(x_sample))
print(type(x_sample[0]))
vertices = create_vertices(x_sample, y_sample, z_sample)

# find_closest_neighbors(vertices)
find_grid_neighbors(vertices)
graph = create_graph(vertices)




start_id = 0
end_id = sample_size-1
path, total_weight = find_shortest_path(graph, start_id, end_id)
# path_vert, total_weight_vert = find_shortest_path_vertical(graph, start_id, end_id)
path_vert, total_weight_vert = find_longest_path(graph, start_id, end_id)              # Takes a very long time
path_hori, total_weight_hori = find_shortest_path_horizontal(graph, start_id, end_id)

print(f"\nShortest path from {start_id} to {end_id}: {path}")
print(f"Total weight of the path: {total_weight:.2f}")

print(f"\nShortest path from {start_id} to {end_id}: {path_vert}")
print(f"Total weight of the path: {total_weight_vert:.2f}")

print(f"\nShortest path from {start_id} to {end_id}: {path_hori}")
print(f"Total weight of the path: {total_weight_hori:.2f}")

print(len(graph.vertices))
plot_surface_with_edges(graph.vertices, path, path_vert, path_hori)




# Basic plot for plotting all of the points (you can increase the resolution significantly)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x_sample, y_sample, z_sample, c=z_sample, cmap="terrain", s=1)
# ax.set_title("3D Topographic Map")
# ax.set_xlabel("X Coordinate")
# ax.set_ylabel("Y Coordinate")
# ax.set_zlabel("Elevation (m)")
# fig.colorbar(scatter, ax=ax, label="Elevation (m)")
# ax.legend()
# plt.show()
