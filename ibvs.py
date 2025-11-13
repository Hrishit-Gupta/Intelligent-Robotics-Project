"""
IBVS Visual Servoing - 3D Simulation with Real-Time Feature Perception
Left: 3D agent motion tracking feature points
Right: Agent's current view of feature points (dynamic image perception)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

plt.ion()  # Enable interactive mode

def skew(v):
    """Create skew-symmetric matrix from vector"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def interaction_matrix_point(x, y, Z):
    """Compute interaction matrix L_x for a single point (Equation 11)"""
    L = np.array([[-1/Z, 0, x/Z, x*y, -(1+x**2), y],
                  [0, -1/Z, y/Z, 1+y**2, -x*y, -x]])
    return L

def interaction_matrix_full(points_2d, depths):
    """Stack interaction matrices for multiple points"""
    n_points = len(points_2d)
    L = np.zeros((2*n_points, 6))
    for i in range(n_points):
        x, y = points_2d[i]
        Z = depths[i]
        L[2*i:2*i+2, :] = interaction_matrix_point(x, y, Z)
    return L

def rotation_matrix(rx, ry, rz):
    """Create rotation matrix from Euler angles (radians)"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def project_points(points_world, t_cam, R_cam):
    """Project 3D points to camera image plane"""
    points_cam = (R_cam.T @ (points_world - t_cam).T).T
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    points_2d = np.column_stack([x, y])
    depths = points_cam[:, 2]
    return points_2d, depths

def draw_camera_frame(ax, t, R, scale=0.08):
    """Draw camera coordinate frame"""
    origin = t
    x_axis = origin + R @ np.array([scale, 0, 0])
    y_axis = origin + R @ np.array([0, scale, 0])
    z_axis = origin + R @ np.array([0, 0, scale])
    
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
            'r-', linewidth=3, alpha=0.8)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
            'g-', linewidth=3, alpha=0.8)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
            'b-', linewidth=3, alpha=0.8)

# ============================================================================
# SETUP
# ============================================================================

# Define 3D square target (ArUco-like markers)
square_size = 0.2
z_target = 0.5

points_3d = np.array([
    [-square_size/2, -square_size/2, z_target],
    [square_size/2, -square_size/2, z_target],
    [square_size/2, square_size/2, z_target],
    [-square_size/2, square_size/2, z_target]
])

# Desired camera pose
t_desired = np.array([0.0, 0.0, 0.0])
R_desired = np.eye(3)

# Initial camera pose (larger displacement)
t_initial = np.array([0.15, -0.12, 0.2])
R_initial = rotation_matrix(np.deg2rad(25), np.deg2rad(-20), np.deg2rad(30))

# Get desired features
s_desired, depths_desired = project_points(points_3d, t_desired, R_desired)

# ============================================================================
# SHOW DESIRED IMAGE FIRST
# ============================================================================

print("="*70)
print("SHOWING DESIRED FEATURE CONFIGURATION")
print("="*70)

fig_desired = plt.figure(figsize=(8, 8))
ax_desired = fig_desired.add_subplot(111)
ax_desired.set_xlim(-0.3, 0.3)
ax_desired.set_ylim(-0.3, 0.3)
ax_desired.set_aspect('equal')
ax_desired.grid(True, alpha=0.3)
ax_desired.set_xlabel('x (normalized)', fontsize=14, fontweight='bold')
ax_desired.set_ylabel('y (normalized)', fontsize=14, fontweight='bold')
ax_desired.set_title('Desired Feature Points Configuration\n(Target for IBVS Control)', 
                     fontsize=16, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
marker_labels = ['Bottom-Left', 'Bottom-Right', 'Top-Right', 'Top-Left']

for i in range(4):
    ax_desired.scatter(s_desired[i, 0], s_desired[i, 1], 
                      marker=markers[i], color=colors[i], s=400,
                      edgecolors='black', linewidths=3, 
                      label=f'Point {i+1} ({marker_labels[i]})', zorder=5)

# Draw square connecting lines
square_order = [0, 1, 2, 3, 0]
for i in range(4):
    ax_desired.plot([s_desired[square_order[i], 0], s_desired[square_order[i+1], 0]],
                   [s_desired[square_order[i], 1], s_desired[square_order[i+1], 1]],
                   'k--', linewidth=2, alpha=0.5)

ax_desired.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.show(block=True)  # Block until this window is closed

print("\nDesired image closed. Starting IBVS simulation...")
print("="*70)


# IBVS CONTROL SIMULATION


lambda_gain = 0.2 
dt = 0.5  
max_iterations = 400  
threshold = 1e-4

# Initialize
t_current = t_initial.copy()
R_current = R_initial.copy()
s_desired_vec = s_desired.flatten()

# Storage for animation
all_positions = []
all_rotations = []
all_features = []
all_errors = []

converged_flag = False
convergence_iter = max_iterations

# Run IBVS control loop
print("\nRunning IBVS Control Loop...")
for iteration in range(max_iterations):
    s_current, depths_current = project_points(points_3d, t_current, R_current)
    
    if np.any(depths_current <= 0.01):
        print(f"Warning: Points behind camera at iteration {iteration}")
        break
    
    # Store current state
    all_positions.append(t_current.copy())
    all_rotations.append(R_current.copy())
    all_features.append(s_current.copy())
    
    # Compute error
    s_current_vec = s_current.flatten()
    e = s_current_vec - s_desired_vec
    error_norm = np.linalg.norm(e)
    all_errors.append(error_norm)
    
    # Check convergence
    if error_norm < threshold and not converged_flag:
        print(f"✓ Converged at iteration {iteration}, error = {error_norm:.2e}")
        converged_flag = True
        convergence_iter = iteration
    
    if converged_flag and iteration > convergence_iter + 10:
        break
    
    # Compute interaction matrices
    L_current = interaction_matrix_full(s_current, depths_current)
    L_desired = interaction_matrix_full(s_desired, depths_desired)
    L_avg = 0.5 * (L_current + L_desired)
    L_avg_pinv = np.linalg.pinv(L_avg)
    
    # Control law: v_c = -λ * L̂⁺_e * e
    v_c = -lambda_gain * L_avg_pinv @ e
    
    v_cam = v_c[0:3]
    omega_cam = v_c[3:6]
    
    # Update pose
    v_world = R_current @ v_cam
    t_current = t_current + v_world * dt
    
    omega_norm = np.linalg.norm(omega_cam)
    if omega_norm > 1e-10:
        theta = omega_norm * dt
        k = omega_cam / omega_norm
        K = skew(k)
        R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        R_current = R_current @ R_delta
    
    if iteration % 30 == 0:
        print(f"Iter {iteration:3d}: error = {error_norm:.4e}")

all_positions = np.array(all_positions)
num_frames = len(all_positions)

print(f"\nSimulation complete: {num_frames} frames generated")
print(f"Final error: {all_errors[-1]:.2e}")
print("="*70)

# ============================================================================
# ANIMATED VISUALIZATION WITH MANUAL UPDATE
# ============================================================================

print("\nCreating animation window...")

fig = plt.figure(figsize=(18, 8))

# Left: 3D scene
ax_3d = fig.add_subplot(121, projection='3d')

# Right: Current image perception
ax_img = fig.add_subplot(122)

# Setup 3D plot
ax_3d.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax_3d.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax_3d.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
ax_3d.set_title('3D Agent Motion (IBVS Control)', fontsize=13, fontweight='bold')

# Set 3D limits
all_x = np.concatenate([all_positions[:, 0], points_3d[:, 0]])
all_y = np.concatenate([all_positions[:, 1], points_3d[:, 1]])
all_z = np.concatenate([all_positions[:, 2], points_3d[:, 2]])
margin = 0.15
ax_3d.set_xlim([all_x.min() - margin, all_x.max() + margin])
ax_3d.set_ylim([all_y.min() - margin, all_y.max() + margin])
ax_3d.set_zlim([0, all_z.max() + margin])
ax_3d.view_init(elev=20, azim=45)

# Draw feature points (ArUco markers) - static
for i in range(4):
    ax_3d.scatter(points_3d[i, 0], points_3d[i, 1], points_3d[i, 2],
                 marker=markers[i], color=colors[i], s=200, 
                 edgecolors='black', linewidths=2, label=f'Point {i+1}')

# Draw target square - static
for i in range(4):
    ax_3d.plot([points_3d[square_order[i], 0], points_3d[square_order[i+1], 0]],
              [points_3d[square_order[i], 1], points_3d[square_order[i+1], 1]],
              [points_3d[square_order[i], 2], points_3d[square_order[i+1], 2]],
              'k-', linewidth=2, alpha=0.6)

ax_3d.legend(fontsize=9, loc='upper right')

# Setup image plot
ax_img.set_xlim(-0.35, 0.35)
ax_img.set_ylim(-0.35, 0.35)
ax_img.set_aspect('equal')
ax_img.grid(True, alpha=0.3)
ax_img.set_xlabel('x (normalized)', fontsize=12, fontweight='bold')
ax_img.set_ylabel('y (normalized)', fontsize=12, fontweight='bold')
ax_img.set_title("Agent's Current Perception", fontsize=13, fontweight='bold')

# Plot desired features (static reference)
for i in range(4):
    ax_img.scatter(s_desired[i, 0], s_desired[i, 1], 
                  marker=markers[i], s=300, facecolors='none',
                  edgecolors=colors[i], linewidths=3, alpha=0.5, zorder=3)

# Draw desired square
for i in range(4):
    ax_img.plot([s_desired[square_order[i], 0], s_desired[square_order[i+1], 0]],
               [s_desired[square_order[i], 1], s_desired[square_order[i+1], 1]],
               'gray', linestyle='--', linewidth=1.5, alpha=0.5)

plt.suptitle('IBVS Visual Servoing: Agent Motion & Feature Perception', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

print("Animation starting... (Press Ctrl+C to stop)")
print("="*70)

# Animation loop with manual frame updates
try:
    skip_frames = max(1, num_frames // 200)  # Show ~200 frames max
    
    # Store handles for dynamic elements
    trajectory_line = None
    agent_scatter = None
    camera_frame_lines = []
    current_feature_scatters = []
    current_square_lines = []
    text_annotation = None
    
    for frame in range(0, num_frames, skip_frames):
        # Remove previous dynamic elements from 3D plot
        if trajectory_line:
            trajectory_line.remove()
        if agent_scatter:
            agent_scatter.remove()
        for line in camera_frame_lines:
            line.remove()
        camera_frame_lines = []
        
        # Remove previous dynamic elements from image plot
        for scatter in current_feature_scatters:
            scatter.remove()
        current_feature_scatters = []
        
        for line in current_square_lines:
            line.remove()
        current_square_lines = []
        
        if text_annotation:
            text_annotation.remove()
        
        # Draw 3D trajectory up to current frame
        if frame > 0:
            trajectory_line, = ax_3d.plot(all_positions[:frame+1, 0], 
                                          all_positions[:frame+1, 1], 
                                          all_positions[:frame+1, 2], 
                                          'b-', linewidth=2.5, alpha=0.7)
        
        # Draw current agent position
        t = all_positions[frame]
        R = all_rotations[frame]
        agent_scatter = ax_3d.scatter([t[0]], [t[1]], [t[2]], c='red', s=200, marker='o', 
                                     edgecolors='black', linewidths=2.5, zorder=10)
        
        # Draw camera frame
        origin = t
        x_axis = origin + R @ np.array([0.08, 0, 0])
        y_axis = origin + R @ np.array([0, 0.08, 0])
        z_axis = origin + R @ np.array([0, 0, 0.08])
        
        line_x, = ax_3d.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
                             'r-', linewidth=3, alpha=0.8)
        line_y, = ax_3d.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
                             'g-', linewidth=3, alpha=0.8)
        line_z, = ax_3d.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
                             'b-', linewidth=3, alpha=0.8)
        camera_frame_lines = [line_x, line_y, line_z]
        
        # Draw current feature perception
        s_curr = all_features[frame]
        for i in range(4):
            scatter = ax_img.scatter(s_curr[i, 0], s_curr[i, 1], 
                                    marker=markers[i], color=colors[i], s=350,
                                    edgecolors='black', linewidths=2.5, zorder=5)
            current_feature_scatters.append(scatter)
        
        # Draw current square
        for i in range(4):
            line, = ax_img.plot([s_curr[square_order[i], 0], s_curr[square_order[i+1], 0]],
                               [s_curr[square_order[i], 1], s_curr[square_order[i+1], 1]],
                               'k-', linewidth=2.5, alpha=0.8)
            current_square_lines.append(line)
        
        # Add text annotations
        err = all_errors[frame]
        text_annotation = ax_img.text(0.02, 0.98, 
                                     f'Error: {err:.4f}\nIter: {frame}\nConverged: {"Yes" if frame >= convergence_iter else "No"}', 
                                     transform=ax_img.transAxes, fontsize=11, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.draw()
        plt.pause(0.05)  # 50ms pause between frames
        
        # Break if converged and shown enough frames after
        if frame > convergence_iter + 20:
            break

except KeyboardInterrupt:
    print("\nAnimation stopped by user")

print("\nAnimation complete. Close the window to finish.")
plt.show(block=True)

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"Final error: {all_errors[-1]:.2e}")
print(f"Convergence: {'SUCCESS' if converged_flag else 'PARTIAL'}")
print(f"Total iterations shown: {num_frames}")
print("="*70)