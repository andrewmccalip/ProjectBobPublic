import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
from scipy.spatial import ConvexHull
import pandas as pd
from copy import deepcopy


#user inputs 
stl_file = 'hull_analysis\hull9.stl'
total_mass = 210  #pounds
cog_x = -.01   #from CAD assembly        
cog_y = -2  #from CAD assembly   
cog_z = -77    #from CAD assembly
    
    

def read_stl(file_path):
    stl_mesh = mesh.Mesh.from_file(file_path)
    return stl_mesh

def calculate_total_volume(stl_mesh):
    volume = 0
    for i in range(len(stl_mesh.vectors)):
        v0, v1, v2 = stl_mesh.vectors[i]
        volume += np.dot(v0, np.cross(v1, v2)) / 6
    return abs(volume)

def calculate_submerged_volume_and_centroid(stl_mesh, waterline_height):
    # Convert the STL mesh to a Trimesh object
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Slice the mesh at the waterline height
    slice_plane = trimesh.intersections.slice_mesh_plane(trimesh_mesh, plane_normal=[0, -1, 0], plane_origin=[0, waterline_height, 0])
    
    # Calculate the volume and centroid of the submerged part
    submerged_volume = slice_plane.volume
    submerged_centroid = slice_plane.center_mass
    
    return submerged_volume, submerged_centroid

def find_waterline_for_buoyancy(stl_mesh, total_mass, density_seawater=64):
    required_volume = total_mass / density_seawater * 1728  # Convert cubic feet to cubic inches
    print(f"Required submerged volume: {required_volume:.2f} cubic inches")
    
    waterline_height = -2
    step = 0.03  # Move in the negative Y direction
    tolerance = 100
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        submerged_volume, _ = calculate_submerged_volume_and_centroid(stl_mesh, waterline_height)
        #print(f"Iteration {iteration}: Waterline Height = {waterline_height:.2f} units, Submerged Volume = {submerged_volume:.2f} cubic units")
        if abs(submerged_volume - required_volume) < tolerance:
            break
        if submerged_volume < required_volume:
            waterline_height += step
        else:
            waterline_height -= step
        iteration += 1
    
    return waterline_height

def plot_centers_and_outline_xy(stl_mesh, center_of_mass, center_of_buoyancy, waterline_height):
    plt.figure(figsize=(10, 6))
    
    # Get all vertices from the STL mesh
    vertices = stl_mesh.vectors.reshape(-1, 3)
    
    # Extract x and y coordinates
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Plot all vertices
    plt.scatter(x, y, color='gray', s=1, alpha=0.5, label='Hull vertices')
    
    # Plot the centers of mass and buoyancy
    plt.scatter(center_of_mass[0], center_of_mass[1], color='red', label='Center of Mass')
    plt.scatter(center_of_buoyancy[0], center_of_buoyancy[1], color='blue', label='Center of Buoyancy')
    
    # Plot the waterline
    plt.axhline(y=waterline_height, color='lightblue', linestyle='--', label='Waterline')
    
    plt.xlabel('X-axis (inches)')
    plt.ylabel('Y-axis (inches)')
    plt.title('Centers of Mass and Buoyancy with Hull Vertices (XY Plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Calculate the angles of each point with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Sort the points based on their angles
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def plot_centers_and_outline_yz(stl_mesh, center_of_mass, center_of_buoyancy, waterline_height):
    plt.figure(figsize=(10, 6))
    
    # Project the 3D vertices onto the YZ plane
    yz_projection = stl_mesh.vectors[:, :, [1, 2]].reshape(-1, 2)
    
    # Plot the outline of the hull
    hull = ConvexHull(yz_projection)
    for simplex in hull.simplices:
        plt.plot(yz_projection[simplex, 1], yz_projection[simplex, 0], 'k-')  # Swap axes for plotting
    
    # Plot the centers of mass and buoyancy
    plt.scatter(center_of_mass[2], center_of_mass[1], color='red', label='Center of Mass')  # Swap axes for plotting
    plt.scatter(center_of_buoyancy[2], center_of_buoyancy[1], color='blue', label='Center of Buoyancy')  # Swap axes for plotting
    
    # Plot the waterline
    plt.axvline(x=waterline_height, color='lightblue', linestyle='--', label='Waterline')
    
    plt.xlabel('Z-axis (inches)')
    plt.ylabel('Y-axis (inches)')
    plt.title('Centers of Mass and Buoyancy with Hull Outline (YZ Plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_3d_stl(stl_mesh, waterline_height):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    faces_above = []
    faces_below = []
    
    for triangle in stl_mesh.vectors:
        y_values = triangle[:, 1]
        if np.all(y_values <= waterline_height):
            faces_below.append(triangle)
        else:
            faces_above.append(triangle)
    
    ax.add_collection3d(Poly3DCollection(faces_above, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(faces_below, facecolors='blue', linewidths=1, edgecolors='r', alpha=.25))
    
    x_limits = [stl_mesh.x.min(), stl_mesh.x.max()]
    y_limits = [stl_mesh.y.min(), stl_mesh.y.max()]
    z_limits = [stl_mesh.z.min(), stl_mesh.z.max()]
    
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    
    max_range = np.array([np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)]).max() / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range)
    ax.set_zlim(mid_z - max_range)
    
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    plt.show()

def rotate_mesh_around_cg(stl_mesh, center_of_mass, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    
    # Translate mesh so that CG is at origin
    stl_mesh.vectors -= center_of_mass
    
    # Rotate mesh
    stl_mesh.vectors = np.dot(stl_mesh.vectors.reshape(-1, 3), rotation_matrix).reshape(stl_mesh.vectors.shape)
    
    # Translate mesh back
    stl_mesh.vectors += center_of_mass
    
    return stl_mesh, center_of_mass  # CG doesn't change

def stability_plot():
   
    # Define parameters for stability analysis# Define parameters for stability analysis
    max_angle = 90
    min_angle = -90
    angle_step = 4
    
    stl_mesh = read_stl(stl_file)
    center_of_mass = np.array([cog_x, cog_y, cog_z])
    
    
    angles = np.arange(min_angle, max_angle + angle_step, angle_step)
    righting_moments = []
    stability_cases = []
    colors = []

    for roll_angle in angles:
        print(f"Processing angle: {roll_angle} degrees")
        
        # Store original mesh for comparison
        original_mesh = deepcopy(stl_mesh)
        
        # Rotate the mesh around the CG
        rotated_mesh, rotated_center_of_mass = rotate_mesh_around_cg(deepcopy(stl_mesh), center_of_mass, roll_angle)
        
        # Calculate waterline and center of buoyancy for the rotated mesh
        waterline_height = find_waterline_for_buoyancy(rotated_mesh, total_mass)
        submerged_volume, center_of_buoyancy = calculate_submerged_volume_and_centroid(rotated_mesh, waterline_height)
        
        # Calculate righting moment
        righting_moment = (rotated_center_of_mass[0] - center_of_buoyancy[0]) * total_mass
        righting_moments.append(righting_moment)
        
        if (roll_angle > 0 and righting_moment < 0) or (roll_angle < 0 and righting_moment > 0):
            stability_cases.append('Stable')
            colors.append('green')
        else:
            stability_cases.append('Unstable')
            colors.append('red')

    df = pd.DataFrame({
        'Angle (degrees)': angles, 
        'Righting Moment (pound-inches)': righting_moments,
        'Stability Case': stability_cases,
        'Color': colors
    })
    print(df)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Angle (degrees)'], df['Righting Moment (pound-inches)'], 
                          c=df['Color'], marker='o')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Righting Moment (pound-inches)')
    plt.title('Righting Moment vs. Angle of Rotation')
    plt.grid(True)
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Stable',
                                  markerfacecolor='green', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='Unstable',
                                  markerfacecolor='red', markersize=10)]
    plt.legend(handles=legend_elements)
    
    # Add CG and mass information to the lower left corner
    cg_info = f'CG: ({cog_x:.2f}, {cog_y:.2f}, {cog_z:.2f})\nMass: {total_mass} lbs'
    plt.text(0.02, 0.02, cg_info, transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()

def single_angle():
   
  
  

    stl_mesh = read_stl(stl_file)
    center_of_mass = np.array([cog_x, cog_y, cog_z])

    roll_angle = 30  # Single angle for analysis
    print(f"Processing angle: {roll_angle} degrees")
    
    # Store original mesh for comparison
    original_mesh = deepcopy(stl_mesh)
    
    # Rotate the mesh around the CG
    rotated_mesh, center_of_mass = rotate_mesh_around_cg(deepcopy(stl_mesh), center_of_mass, roll_angle)
    
    # Calculate waterline and center of buoyancy for the rotated mesh
    waterline_height = find_waterline_for_buoyancy(rotated_mesh, total_mass)
    submerged_volume, center_of_buoyancy = calculate_submerged_volume_and_centroid(rotated_mesh, waterline_height)
    
    # Calculate righting moment
    righting_moment = (center_of_mass[0] - center_of_buoyancy[0]) * total_mass

    print(f"Center of Mass: {center_of_mass}")
    print(f"Waterline Height: {waterline_height}")
    print(f"Submerged Volume: {submerged_volume}")
    print(f"Center of Buoyancy: {center_of_buoyancy}")
    print(f"Righting Moment: {righting_moment}")

    # Plot both original and rotated mesh
    plot_comparison(original_mesh, rotated_mesh, center_of_mass, center_of_mass, center_of_buoyancy, waterline_height)

def plot_comparison(original_mesh, rotated_mesh, original_cg, rotated_cg, center_of_buoyancy, waterline_height):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original mesh
    ax1.scatter(original_mesh.vectors[:, :, 0].flatten(), original_mesh.vectors[:, :, 1].flatten(), color='gray', s=1, alpha=0.5)
    ax1.scatter(original_cg[0], original_cg[1], color='red', s=100, label='Original CG')
    ax1.set_title('Original Mesh')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True)
    
    # Plot rotated mesh
    ax2.scatter(rotated_mesh.vectors[:, :, 0].flatten(), rotated_mesh.vectors[:, :, 1].flatten(), color='gray', s=1, alpha=0.5)
    ax2.scatter(rotated_cg[0], rotated_cg[1], color='blue', s=100, label='Rotated CG')
    ax2.scatter(center_of_buoyancy[0], center_of_buoyancy[1], color='green', s=100, label='Center of Buoyancy')
    ax2.axhline(y=waterline_height, color='cyan', linestyle='--', label='Waterline')
    ax2.set_title('Rotated Mesh')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True)
    
    # Set the same limits for both plots
    xlim = ylim = max(
        abs(original_mesh.vectors[:, :, 0].max()),
        abs(original_mesh.vectors[:, :, 1].max()),
        abs(rotated_mesh.vectors[:, :, 0].max()),
        abs(rotated_mesh.vectors[:, :, 1].max())
    )
    ax1.set_xlim(-xlim, xlim)
    ax1.set_ylim(-ylim, ylim)
    ax2.set_xlim(-xlim, xlim)
    ax2.set_ylim(-ylim, ylim)
    
    plt.show()


if __name__ == "__main__":
    single_angle()
    stability_plot()
   