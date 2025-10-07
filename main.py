"""
IBVS with Average Interaction Matrix
Target in front of camera with positive depths
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    """
    Project 3D points to camera image plane
    
    points_world: 3D points in world frame (N x 3)
    t_cam: camera position in world frame (3,)
    R_cam: camera orientation in world frame (3 x 3)
    
    Returns: normalized image coordinates (N x 2), depths (N,)
    """
    # Transform points from world to camera frame
    # Standard formula: P_cam = R_cam.T @ (P_world - t_cam)
    points_cam = (R_cam.T @ (points_world - t_cam).T).T
    
    # Perspective projection
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    
    points_2d = np.column_stack([x, y])
    depths = points_cam[:, 2]
    
    return points_2d, depths


# CORRECTED GEOMETRIC SETUP


# Define 3D square target IN FRONT OF CAMERA at Z = 0.5m
square_size = 0.2  # 20 cm
z_target = 0.5  # Target 50cm in front of camera

points_3d = np.array([
    [-square_size/2, -square_size/2, z_target],
    [square_size/2, -square_size/2, z_target],
    [square_size/2, square_size/2, z_target],
    [-square_size/2, square_size/2, z_target]
])

# Desired camera pose: at origin, looking down +Z axis
t_desired = np.array([0.0, 0.0, 0.0])
R_desired = np.eye(3)  # Identity = looking along +Z

# Initial camera pose: small displacement from desired
t_initial = np.array([0.05, -0.03, 0.08])  # Shifted slightly
R_initial = rotation_matrix(np.deg2rad(10), np.deg2rad(-12), np.deg2rad(15))

print("="*70)
print("GEOMETRIC SETUP VERIFICATION")
print("="*70)
print(f"Target square center at: [0, 0, {z_target}] (world frame)")
print(f"Target is {z_target*100:.1f} cm in front of camera origin")
print()

# Verify desired pose
s_desired, depths_desired = project_points(points_3d, t_desired, R_desired)
print("DESIRED POSE:")
print(f"  Camera at: {t_desired}")
print(f"  Target depths: {depths_desired}")
print(f"  All positive: {np.all(depths_desired > 0)} " if np.all(depths_desired > 0) else "  ERROR: negative depths")
print()

# Verify initial pose
s_initial, depths_initial = project_points(points_3d, t_initial, R_initial)
print("INITIAL POSE:")
print(f"  Camera at: {t_initial}")
print(f"  Target depths: {depths_initial}")
print(f"  All positive: {np.all(depths_initial > 0)} " if np.all(depths_initial > 0) else "  ERROR: negative depths")
print(f"  Displacement from desired: {np.linalg.norm(t_initial - t_desired)*100:.2f} cm")
print("="*70)

if not (np.all(depths_desired > 0) and np.all(depths_initial > 0)):
    print("\nERROR: Setup still incorrect. Exiting.")
    import sys
    sys.exit(1)


# CONTROL PARAMETERS

lambda_gain = 0.5
dt = 1.0
max_iterations = 150
threshold = 1e-4

# Flatten desired features
s_desired_vec = s_desired.flatten()

# Storage
trajectory_image = [[] for _ in range(4)]
trajectory_3d = []
error_history = []
velocity_history = []

# Current pose
t_current = t_initial.copy()
R_current = R_initial.copy()


# VISUAL SERVO CONTROL LOOP


print("\nStarting IBVS control loop...")
print(f"Control gain λ = {lambda_gain}, Time step dt = {dt}")
print("="*70)

converged = False

for iteration in range(max_iterations):
    # Project points to get current features
    s_current, depths_current = project_points(points_3d, t_current, R_current)
    
    # Safety check
    if np.any(depths_current <= 0.01):
        print(f"\nWARNING: Points behind camera at iteration {iteration}")
        print(f"Camera at: {t_current}")
        print(f"Depths: {depths_current}")
        break
    
    # Store trajectory
    for i in range(4):
        trajectory_image[i].append(s_current[i].copy())
    trajectory_3d.append(t_current.copy())
    
    # Compute error
    s_current_vec = s_current.flatten()
    e = s_current_vec - s_desired_vec
    error_norm = np.linalg.norm(e)
    error_history.append(error_norm)
    
    # Check convergence
    if error_norm < threshold:
        print(f"\n CONVERGED at iteration {iteration}")
        print(f"  Final error: {error_norm:.2e}")
        converged = True
        break
    
    # Compute interaction matrices
    L_current = interaction_matrix_full(s_current, depths_current)
    L_desired = interaction_matrix_full(s_desired, depths_desired)
    
    # Average interaction matrix (Figure 4 method)
    L_avg = 0.5 * (L_current + L_desired)
    
    # Compute pseudoinverse
    L_avg_pinv = np.linalg.pinv(L_avg)
    
    # Control law (Equation 5): v_c = -λ * L̂⁺_e * e
    v_c = -lambda_gain * L_avg_pinv @ e
    velocity_history.append(v_c.copy())
    
    # Extract velocities (in camera frame)
    v_cam = v_c[0:3]
    omega_cam = v_c[3:6]
    
    # Transform linear velocity to world frame
    v_world = R_current @ v_cam
    
    # Update camera position
    t_current = t_current + v_world * dt
    
    # Update camera orientation
    omega_norm = np.linalg.norm(omega_cam)
    if omega_norm > 1e-10:
        theta = omega_norm * dt
        k = omega_cam / omega_norm
        K = skew(k)
        R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        R_current = R_current @ R_delta
    
    # Progress
    if iteration % 20 == 0:
        print(f"Iter {iteration:3d}: error = {error_norm:.4e}, "
              f"|v| = {np.linalg.norm(v_c[:3])*100:.2f} cm/s, "
              f"|ω| = {np.rad2deg(np.linalg.norm(v_c[3:])):.2f} deg/s")

if not converged:
    print(f"\nDid not fully converge within {max_iterations} iterations")
    if len(error_history) > 0:
        print(f"Final error: {error_history[-1]:.4e}")

print(f"\nFinal camera position: {t_current}")
print(f"Position error: {np.linalg.norm(t_current - t_desired)*100:.3f} cm")
print("="*70)

# Convert to arrays
trajectory_3d = np.array(trajectory_3d)
velocity_history = np.array(velocity_history)
error_history = np.array(error_history)


# VISUALIZATION - Figure 4 Style


fig = plt.figure(figsize=(16, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# (a) Image trajectories
ax1 = plt.subplot(2, 2, 1)
for i in range(4):
    traj = np.array(trajectory_image[i])
    ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2.5, 
             alpha=0.8, label=f'Point {i+1}')
    ax1.scatter(traj[0, 0], traj[0, 1], marker=markers[i], color=colors[i], 
               s=100, edgecolors='black', linewidths=2, zorder=5)
    ax1.scatter(s_desired[i, 0], s_desired[i, 1], marker=markers[i], 
               color=colors[i], s=150, facecolors='none', 
               edgecolors=colors[i], linewidths=3, zorder=5)

ax1.set_xlabel('x (normalized)', fontsize=12, fontweight='bold')
ax1.set_ylabel('y (normalized)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Image Point Trajectories', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='best')
ax1.axis('equal')

# (b) Linear velocities
ax2 = plt.subplot(2, 2, 2)
iters = range(len(velocity_history))
ax2.plot(iters, velocity_history[:, 0] * 100, 'b-', linewidth=2, label='vx')
ax2.plot(iters, velocity_history[:, 1] * 100, 'g-', linewidth=2, label='vy')
ax2.plot(iters, velocity_history[:, 2] * 100, 'r-', linewidth=2, label='vz')
ax2.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.3)
ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax2.set_ylabel('Linear Velocity (cm/s)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Camera Linear Velocities', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# (c) Angular velocities
ax3 = plt.subplot(2, 2, 3)
ax3.plot(iters, np.rad2deg(velocity_history[:, 3]), 'b-', linewidth=2, label='ωx')
ax3.plot(iters, np.rad2deg(velocity_history[:, 4]), 'g-', linewidth=2, label='ωy')
ax3.plot(iters, np.rad2deg(velocity_history[:, 5]), 'r-', linewidth=2, label='ωz')
ax3.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.3)
ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax3.set_ylabel('Angular Velocity (deg/s)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Camera Angular Velocities', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# (d) 3D trajectory
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot(trajectory_3d[:, 0] * 100, trajectory_3d[:, 1] * 100, 
         trajectory_3d[:, 2] * 100, 'b-', linewidth=3, alpha=0.8)
ax4.scatter([trajectory_3d[0, 0] * 100], [trajectory_3d[0, 1] * 100], 
           [trajectory_3d[0, 2] * 100], c='green', s=200, marker='o', 
           label='Initial', edgecolors='black', linewidths=2)
ax4.scatter([t_desired[0] * 100], [t_desired[1] * 100], 
           [t_desired[2] * 100], c='red', s=300, marker='*', 
           label='Desired', edgecolors='black', linewidths=2)
ax4.scatter([0], [0], [z_target * 100], c='orange', s=150, marker='s', 
           label='Target', edgecolors='black', linewidths=2)
ax4.set_xlabel('X (cm)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Y (cm)', fontsize=11, fontweight='bold')
ax4.set_zlabel('Z (cm)', fontsize=11, fontweight='bold')
ax4.set_title('(d) 3D Camera Trajectory', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.view_init(elev=20, azim=45)

plt.suptitle('IBVS with Average Interaction Matrix: L̂⁺ₑ = ½(Lₑ + L*ₑ)⁺', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('IBVS_Control.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as 'figure_4_correct_geometry.png'")
plt.show()

# Error convergence
fig2, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(error_history, 'b-', linewidth=3)
ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('Error Norm (log scale)', fontsize=13, fontweight='bold')
ax.set_title('Error Convergence (Exponential Decrease)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('error_convergence.png', dpi=300, bbox_inches='tight')
plt.show()


# FINAL SUMMARY

print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)
print(f"Method: Average Interaction Matrix L̂⁺ₑ = ½(Lₑ + L*ₑ)⁺")
print(f"Total iterations: {len(error_history)}")
print(f"Initial error: {error_history[0]:.6f}")
print(f"Final error: {error_history[-1]:.2e}")
print(f"Error reduction: {error_history[0]/error_history[-1]:.1f}x")
print(f"Max linear velocity: {np.max(np.abs(velocity_history[:, :3]))*100:.2f} cm/s")
print(f"Max angular velocity: {np.max(np.abs(np.rad2deg(velocity_history[:, 3:]))):.2f} deg/s")
print(f"Final position error: {np.linalg.norm(t_current - t_desired)*100:.3f} cm")
print(f"Convergence status: {'SUCCESS' if converged else 'PARTIAL'}")
print("="*70)

if converged:
    print("\nSimulation is converged properly")
    