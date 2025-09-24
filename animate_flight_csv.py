import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ----------------------
# Utility functions (Needed for visualization)
# ----------------------
R_EARTH = 6371000.0

def wrap_deg(a):
    """Wraps an angle or an array of angles to the [-180, 180) range."""
    return (a + 180) % 360 - 180

def bearing_deg(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon)*math.cos(rlat2)
    x = math.cos(rlat1)*math.sin(rlat2) - math.sin(rlat1)*math.cos(rlat2)*math.cos(dlon)
    return math.degrees(math.atan2(y, x))

def dist_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R_EARTH*math.atan2(math.sqrt(a), math.sqrt(1-a))

def _to_local_xy(lat0, lon0, latp, lonp):
    dN = (latp - lat0) * (math.pi/180) * R_EARTH
    dE = (lonp - lon0) * (math.pi/180) * R_EARTH * math.cos(math.radians(lat0))
    return dE, dN

# Scoring function
def calculate_landing_score(distance, pad_radius):
    if distance <= 1.0: return 100, "🎯 Bullseye!"
    elif distance <= pad_radius / 2: return 75, "✅ Excellent"
    elif distance <= pad_radius: return 50, "👍 Good (On the Pad)"
    else: return 0, "❌ Missed"

# ----------------------
# Animation from CSV
# ----------------------
def animate_csv_data(csv_filename, 
                     # *** NEW: Parameter for the fixed target coordinates ***
                     target_coords,
                     landing_pad_radius_m=5.0, 
                     box_size_m=2.0,
                     data_hz=1.0,
                     max_frames=1000):
    
    # --- 1. Load Data from CSV ---
    print(f"Loading data from '{csv_filename}'...")
    try:
        data = np.genfromtxt(csv_filename, delimiter=',', names=True)
        lat = data['gps_lat']
        lon = data['gps_lon']
        alt = data['altitude']
        heading = data['heading']
        print(f"Loaded {len(lat)} data points.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None

    # --- 2. Process Data ---
    dt = 1.0 / data_hz
    # The origin is still the first point of the flight
    lat0, lon0 = lat[0], lon[0]
    
    # *** FIX: Use the pre-defined target coordinates instead of the last data point ***
    target_lat, target_lon = target_coords[0], target_coords[1]
    print(f"Using pre-defined target: Lat {target_lat:.6f}, Lon {target_lon:.6f}")

    xs, ys = np.zeros_like(lat), np.zeros_like(lon)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        xs[i], ys[i] = _to_local_xy(lat0, lon0, la, lo)
    xt, yt = _to_local_xy(lat0, lon0, target_lat, target_lon)

    descent_rates = -np.diff(alt, prepend=alt[0]) / dt
    turn_rates = wrap_deg(np.diff(heading, prepend=heading[0])) / dt
    
    # ... (rest of the function is mostly the same) ...
    n = len(xs)
    step = max(1, n // max_frames)
    idxs = np.arange(0, n, step, dtype=int)
    
    xs_d, ys_d, zs_d = xs[idxs], ys[idxs], alt[idxs]
    descent_rates_d = descent_rates[idxs]
    turn_rates_d = turn_rates[idxs]
    heading_d = heading[idxs]
    lat_d, lon_d = lat[idxs], lon[idxs]
    times = np.arange(len(xs_d)) * dt * step

    # --- 3. Create Plots ---
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_xlabel("East (m)"); ax1.set_ylabel("North (m)"); ax1.set_zlabel("Alt (m)")
    ax1.set_title('3D Trajectory (from CSV)')
    
    xmin, xmax = min(xs.min(), xt), max(xs.max(), xt); ymin, ymax = min(ys.min(), yt), max(ys.max(), yt)
    zmin, zmax = 0.0, max(1.0, alt.max()); pad_xy = 0.1 * max(xmax - xmin, ymax - ymin, 1.0)
    ax1.set_xlim(xmin - pad_xy, xmax + pad_xy); ax1.set_ylim(ymin - pad_xy, ymax + pad_xy); ax1.set_zlim(zmin, zmax * 1.1)
    
    gx = np.linspace(xmin - pad_xy, xmax + pad_xy, 15); gy = np.linspace(ymin - pad_xy, ymax + pad_xy, 15)
    GX, GY = np.meshgrid(gx, gy); GZ = np.zeros_like(GX)
    ax1.plot_wireframe(GX, GY, GZ, linewidth=0.3, color='gray')
    ax1.plot([xt], [yt], [0], marker='x', markersize=10, linestyle='None', color='red')
    
    theta = np.linspace(0, 2 * np.pi, 100)
    pad_x = xt + landing_pad_radius_m * np.cos(theta)
    pad_y = yt + landing_pad_radius_m * np.sin(theta)
    ax1.plot(pad_x, pad_y, 0, color='green', linestyle='--', linewidth=1.5)
    traj_line, = ax1.plot([], [], [], lw=1.5, color='blue')
    
    s = box_size_m / 2.0; cube_verts = np.array([[-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],[-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s]])
    edges = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
    cube_lines = [ax1.plot([], [], [], lw=1.2, color='black')[0] for _ in range(len(edges))]
    
    info_t = ax1.text2D(0.02, 0.95, "", transform=ax1.transAxes, fontsize=9)
    dist_t = ax1.text2D(0.02, 0.90, "", transform=ax1.transAxes, fontsize=9, color='green', weight='bold')
    heading_t = ax1.text2D(0.02, 0.85, "", transform=ax1.transAxes, fontsize=9)
    turn_t = ax1.text2D(0.02, 0.80, "", transform=ax1.transAxes, fontsize=9)

    ax2 = fig.add_subplot(222)
    ax2.set_xlabel("East (m)"); ax2.set_ylabel("North (m)"); ax2.set_title('Top-Down View')
    ax2.set_xlim(xmin - pad_xy, xmax + pad_xy); ax2.set_ylim(ymin - pad_xy, ymax + pad_xy)
    ax2.grid(True); ax2.axis('equal')
    # *** FIX: Updated label to be more generic ***
    ax2.scatter(xt, yt, color='r', s=100, label='Target')
    ax2.plot(pad_x, pad_y, color='green', linestyle='--', linewidth=1.5, label=f'{landing_pad_radius_m}m Landing Pad')
    trajectory_line_2d, = ax2.plot([], [], 'b-', alpha=0.7, label='Trajectory')
    current_pos_2d, = ax2.plot([], [], 'ro', markersize=8)
    
    ax3 = fig.add_subplot(223); ax3.set_xlim(0, times[-1]); ax3.set_ylim(0, zmax*1.1); ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)', color='b'); ax3.set_title('Altitude and Descent Rate'); ax3.grid(True)
    ax4 = ax3.twinx(); ax4.set_ylim(min(descent_rates_d.min(), 0) - 0.5, max(descent_rates_d.max(), 3.0) + 1); ax4.set_ylabel('Descent Rate (m/s)', color='r')
    altitude_line, = ax3.plot([], [], 'b-', label='Altitude'); descent_rate_line, = ax4.plot([], [], 'r-', label='Descent Rate')
    
    ax5 = fig.add_subplot(224); ax5.set_xlim(0, times[-1]); ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Heading (deg)', color='g'); ax5.set_title('Heading and Turn Rate'); ax5.grid(True)
    ax5.set_ylim(0, 360)
    ax6 = ax5.twinx(); ax6.set_ylabel('Turn Rate (deg/s)', color='purple')
    ax6.set_ylim(turn_rates_d.min() - 5, turn_rates_d.max() + 5)
    heading_line, = ax5.plot([], [], 'g-', label='Heading')
    turn_rate_line, = ax6.plot([], [], 'purple', linestyle='--', label='Turn Rate')

    ax2.legend(loc='upper right'); ax3.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax4.legend(loc='upper right', bbox_to_anchor=(1, 0.9)); ax5.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax6.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    
    def init(): return []
    
    def update(j):
        # ... (rest of update function is the same)
        traj_line.set_data(xs_d[:j+1], ys_d[:j+1]); traj_line.set_3d_properties(zs_d[:j+1])
        trajectory_line_2d.set_data(xs_d[:j+1], ys_d[:j+1]); current_pos_2d.set_data([xs_d[j]], [ys_d[j]])
        altitude_line.set_data(times[:j+1], zs_d[:j+1]); descent_rate_line.set_data(times[:j+1], descent_rates_d[:j+1])
        heading_line.set_data(times[:j+1], heading_d[:j+1]); turn_rate_line.set_data(times[:j+1], turn_rates_d[:j+1])
        cx, cy, cz = xs_d[j], ys_d[j], zs_d[j]
        for k, (a, b) in enumerate(edges):
            p1 = cube_verts[a] + np.array([cx, cy, cz]); p2 = cube_verts[b] + np.array([cx, cy, cz])
            cube_lines[k].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            cube_lines[k].set_3d_properties([p1[2], p2[2]])
        info_t.set_text(f"Position: x={cx:6.1f} m, y={cy:6.1f} m, z={cz:5.1f} m")
        dist_to_center = math.sqrt((cx - xt)**2 + (cy - yt)**2)
        dist_to_edge = max(0, dist_to_center - landing_pad_radius_m)
        dist_t.set_text(f"Dist to Pad: {dist_to_edge:5.1f} m")
        brg = bearing_deg(lat_d[j], lon_d[j], target_lat, target_lon)
        head_err = wrap_deg(brg - heading_d[j])
        heading_t.set_text(f"Heading: {heading_d[j]:.1f}° (Error: {head_err:.1f}°)")
        turn_t.set_text(f"Turn Rate: {turn_rates_d[j]:.2f}°/s")
        return []

    ani = FuncAnimation(fig, update, frames=len(xs_d), init_func=init, interval=100, blit=False, repeat=False)
    # The final position is now relative to the CSV's end point
    final_x_csv, final_y_csv = xs[-1], ys[-1]
    return ani, fig, {"final_x": final_x_csv, "final_y": final_y_csv, "target_x": xt, "target_y": yt}


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    
    csv_file_to_animate = "flight_log.csv"
    
    # *** NEW: Define your fixed target coordinates here ***
    target_position = [-7.275000, 112.794020]  # [latitude, longitude]

    landing_pad_radius = 5.0
    data_sample_rate_hz = 1.0
    
    print(f"Attempting to animate flight data from '{csv_file_to_animate}'...")
    
    animation, fig, results = animate_csv_data(
        csv_filename=csv_file_to_animate,
        target_coords=target_position, # Pass the fixed target to the function
        landing_pad_radius_m=landing_pad_radius,
        data_hz=data_sample_rate_hz
    )

    if fig:
        plt.tight_layout()
        plt.show()

    # The scoring now reflects the distance to the pre-defined target
    if results:
        # Note: final_distance is now the distance between the CSV's last point
        # and your pre-defined target_position.
        final_distance = math.sqrt((results["final_x"] - results["target_x"])**2 + (results["final_y"] - results["target_y"])**2)
        score, rank = calculate_landing_score(final_distance, landing_pad_radius)

        print("\n--- Landing Results ---")
        print(f"Final distance from target center: {final_distance:.2f} meters")
        print(f"Landing Rank: {rank}")
        print(f"Score: {score}/100")
        print("-----------------------")