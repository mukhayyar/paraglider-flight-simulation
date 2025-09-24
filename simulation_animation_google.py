import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ----------------------
# Utility functions (Unchanged)
# ----------------------
R_EARTH = 6371000.0
def wrap_deg(a: float) -> float:
    while a > 180: a -= 360
    while a < -180: a += 360
    return a
def bearing_deg(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2); dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon)*math.cos(rlat2); x = math.cos(rlat1)*math.sin(rlat2) - math.sin(rlat1)*math.cos(rlat2)*math.cos(dlon)
    return math.degrees(math.atan2(y, x))
def dist_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R_EARTH*math.atan2(math.sqrt(a), math.sqrt(1-a))
def step_latlon(lat, lon, vx, vy, dt):
    dN = vy*dt; dE = vx*dt; dlat = (dN/R_EARTH) * 180/math.pi
    dlon = (dE/(R_EARTH*math.cos(math.radians(lat)))) * 180/math.pi
    return lat + dlat, lon + dlon
def _to_local_xy(lat0, lon0, latp, lonp):
    dN = (latp - lat0) * (math.pi/180) * R_EARTH
    dE = (lonp - lon0) * (math.pi/180) * R_EARTH * math.cos(math.radians(lat0))
    return dE, dN

# ----------------------
# Simulation core (Upgraded with PD Loiter Controller)
# ----------------------
class ParagliderSim:
    def __init__(self,
                 start_lat, start_lon,
                 target_lat, target_lon,
                 wind_speed_ms=5.0, wind_from_deg=270.0,
                 v_trim_ms=10.0,
                 sink_trim_ms=1.2, sink_max_ms=2.5,
                 kp_track=0.08, max_diff=0.8, track_blend=1.0, # Using a responsive kp_track
                 approach_radius_m=60.0, approach_alt_m=40.0,
                 loiter_radius_m=25.0,
                 landing_alt_m=15.0,
                 flare_alt_m=5.0,
                 dt=0.05, sim_time_s=180.0,
                 init_heading_deg=45.0, init_alt_m=500.0):
        self.lat = float(start_lat); self.lon = float(start_lon)
        self.target = (float(target_lat), float(target_lon)); self.dt = float(dt)
        self.sim_time_s = float(sim_time_s)
        self.wind_speed = float(wind_speed_ms); to_dir = (float(wind_from_deg) + 180.0) % 360.0
        self.Wvx = self.wind_speed * math.sin(math.radians(to_dir))
        self.Wvy = self.wind_speed * math.cos(math.radians(to_dir))
        self.v_trim = float(v_trim_ms); self.sink_trim = float(sink_trim_ms)
        self.sink_max = float(sink_max_ms); self.heading = float(init_heading_deg)
        self.alt = float(init_alt_m); self.kp = float(kp_track)
        self.max_diff = float(max_diff); self.track_blend = max(0.0, min(1.0, float(track_blend)))
        self.approach_radius = float(approach_radius_m); self.approach_alt = float(approach_alt_m)
        self.loiter_radius = float(loiter_radius_m)
        self.landing_alt = float(landing_alt_m)
        self.flare_alt = float(flare_alt_m)

        self.INIT, self.CRUISE, self.APPROACH, self.LOITER, self.LANDING, self.FLARE, self.DISARM = range(7)
        self.state = self.CRUISE
        
        # *** NEW: Variable for the loiter PD controller's derivative term ***
        self.last_radius_error = 0.0

        self.left = 0.5; self.right = 0.5; self.brake = 0.25
        self.target_descent_rate = 2.0; self.descent_integral = 0.0; self.last_descent_error = 0.0
        self.KP_DESCENT = 0.1; self.KI_DESCENT = 0.01; self.KD_DESCENT = 0.05
        self.min_control_alt = 0.5; self.heading_deadband = 2.0; self.max_turn_rate = 30.0
        self.turn_rate_gain = 1.2; self.traj_lat, self.traj_lon, self.traj_alt, self.traj_state = [], [], [], []
        self.traj_heading, self.traj_brg, self.traj_dist = [], [], []
        self.traj_left, self.traj_right, self.traj_brake = [], [], []
        self.traj_control_action, self.traj_descent_rate = [], []
        self.traj_diff, self.traj_heading_error, self.traj_turn_rate = [], [], []

    def step(self):
        brg = bearing_deg(self.lat, self.lon, self.target[0], self.target[1])
        d = dist_m(self.lat, self.lon, self.target[0], self.target[1])
        vax = self.v_trim * math.sin(math.radians(self.heading))
        vay = self.v_trim * math.cos(math.radians(self.heading))
        vx = vax + self.Wvx; vy = vay + self.Wvy
        cog = math.degrees(math.atan2(vx, vy))
        
        if self.state == self.CRUISE and (d < self.approach_radius or self.alt < self.approach_alt): self.state = self.APPROACH
        if self.state == self.APPROACH and d < self.loiter_radius:
            if self.alt > self.landing_alt: self.state = self.LOITER
            else: self.state = self.LANDING
        if self.state == self.LOITER and self.alt <= self.landing_alt: self.state = self.LANDING
        if self.state == self.LANDING and self.alt <= self.flare_alt: self.state = self.FLARE
        if self.state == self.FLARE and self.alt <= 0.1: self.state = self.DISARM
        
        control_action = ""; diff = 0.0; e = 0.0
        
        if self.alt > self.min_control_alt:
            if self.state in (self.CRUISE, self.APPROACH):
                e_track = wrap_deg(brg - cog); e_head  = wrap_deg(brg - self.heading)
                e = (self.track_blend * e_track) + ((1.0 - self.track_blend) * e_head)
                e_corrected = 0.0
                if abs(e) > self.heading_deadband: e_corrected = e - np.sign(e) * self.heading_deadband
                diff = max(-self.max_diff, min(self.max_diff, self.kp * e_corrected))
                self.left  = 0.5 - 0.5 * diff; self.right = 0.5 + 0.5 * diff
                if self.state == self.CRUISE: self.brake = 0.25; control_action = "Cruising"
                else: # APPROACH
                    base_brake = 0.30
                    current_sink = self.sink_trim + (self.sink_max - self.sink_trim) * max(0.0, (self.brake - 0.2) / 0.8)
                    error = self.target_descent_rate - current_sink
                    self.descent_integral += error * self.dt; derivative = (error - self.last_descent_error) / self.dt
                    self.last_descent_error = error
                    pid_output = (self.KP_DESCENT*error + self.KI_DESCENT*self.descent_integral + self.KD_DESCENT*derivative)
                    self.brake = np.clip(base_brake + pid_output, 0.0, 1.0)
                    control_action = "Approach"

            elif self.state == self.LOITER:
                # *** MODIFIED: PD Track-based loiter for aggressive wind correction & stability ***
                tangential_track = wrap_deg(brg + 90)

                # Proportional term (P): current error from the circle
                radius_error = d - self.loiter_radius
                
                # Derivative term (D): rate of change of radius error (damping)
                radius_error_derivative = (radius_error - self.last_radius_error) / self.dt
                self.last_radius_error = radius_error

                # PD controller gains for the track correction angle
                kp_radius = 8.0 # Strong proportional response
                kd_radius = 4.0 # Damping to prevent overshoot

                # Combine P and D terms for the full correction
                track_correction = (kp_radius * radius_error) + (kd_radius * radius_error_derivative)
                track_correction = np.clip(track_correction, -75, 75) # Allow for very sharp corrective turns

                # The final desired track over the ground
                desired_track = wrap_deg(tangential_track - track_correction)
                
                # The error is the difference between our desired track and our actual track (COG).
                e = wrap_deg(desired_track - cog)
                
                loiter_kp = self.kp * 1.5 
                diff = max(-self.max_diff, min(self.max_diff, loiter_kp * e))
                
                self.left = 0.5 - 0.5 * diff
                self.right = 0.5 + 0.5 * diff
                
                self.brake = 0.4
                control_action = "Loitering (PD Corrected)"

            elif self.state == self.LANDING:
                self.left = 0.5; self.right = 0.5; self.brake = min(0.8, self.brake + 0.01)
                control_action = "Final Approach"
            elif self.state == self.FLARE:
                self.left = 0.5; self.right = 0.5; self.brake = min(1.0, self.brake + 0.05)
                control_action = "Flare"
        else:
            self.brake = 1.0; self.left = 0.5; self.right = 0.5
            control_action = "Landed"

        brake_diff = self.right - self.left; airspeed_factor = self.v_trim / 10.0
        turn_rate = brake_diff * self.max_turn_rate * self.turn_rate_gain / airspeed_factor
        self.heading = wrap_deg(self.heading + turn_rate * self.dt); base = 0.20
        span = max(1e-6, 1.0 - base); brake_norm = max(0.0, min(1.0, (self.brake - base)/span))
        sink = self.sink_trim + (self.sink_max - self.sink_trim) * brake_norm
        self.alt = max(0.0, self.alt - sink * self.dt)
        self.lat, self.lon = step_latlon(self.lat, self.lon, vx, vy, self.dt)
        self.traj_lat.append(self.lat); self.traj_lon.append(self.lon); self.traj_alt.append(self.alt)
        self.traj_state.append(self.state); self.traj_heading.append(self.heading); self.traj_brg.append(brg)
        self.traj_dist.append(d); self.traj_left.append(self.left); self.traj_right.append(self.right)
        self.traj_brake.append(self.brake); self.traj_control_action.append(control_action)
        self.traj_descent_rate.append(sink); self.traj_diff.append(diff); self.traj_heading_error.append(e)
        self.traj_turn_rate.append(turn_rate)

    def run(self):
        steps = int(self.sim_time_s / self.dt)
        for _ in range(steps):
            if self.state == self.DISARM and self.alt <= 0.1: break
            self.step()
        return {k: np.array(v) for k, v in self.__dict__.items() if k.startswith("traj_")}

# ----------------------
# Animation helpers
# ----------------------
def simulate_and_animate_3d(start_lat, start_lon, target_lat, target_lon,
                            wind_speed_ms=5.0, wind_from_deg=270.0,
                            sim_time_s=60.0, dt=0.1, max_frames=500,
                            kp_track=0.08, max_diff=0.8, track_blend=1.0,
                            box_size_m=2.0, init_alt_m=500.0,
                            landing_pad_radius_m=5.0,
                            loiter_radius_m=25.0,
                            landing_alt_m=15.0,
                            approach_alt_m=40.0):
    
    sim = ParagliderSim(start_lat, start_lon, target_lat, target_lon,
                        wind_speed_ms=wind_speed_ms, wind_from_deg=wind_from_deg,
                        sim_time_s=sim_time_s, dt=dt,
                        kp_track=kp_track, max_diff=max_diff, track_blend=track_blend,
                        init_alt_m=init_alt_m, landing_alt_m=landing_alt_m,
                        approach_alt_m=approach_alt_m, loiter_radius_m=loiter_radius_m)
    result = sim.run()
    result["target"] = sim.target
    lat, lon, alt, target = result["traj_lat"], result["traj_lon"], result["traj_alt"], result["target"]
    if len(lat) == 0: print("Simulation ended with no trajectory data."); return None, None, None
    control_actions, descent_rates = result["traj_control_action"], result["traj_descent_rate"]
    left_controls, right_controls, brake_controls = result["traj_left"], result["traj_right"], result["traj_brake"]
    diff_values, heading_errors, turn_rates = result["traj_diff"], result["traj_heading_error"], result["traj_turn_rate"]
    lat0, lon0 = lat[0], lon[0]; xs, ys = np.zeros_like(lat), np.zeros_like(lon)
    for i, (la, lo) in enumerate(zip(lat, lon)): xs[i], ys[i] = _to_local_xy(lat0, lon0, la, lo)
    xt, yt = _to_local_xy(lat0, lon0, target[0], target[1]); n = len(xs); step = max(1, n // max_frames)
    idxs = np.arange(0, n, step, dtype=int); xs_d, ys_d, zs_d = xs[idxs], ys[idxs], alt[idxs]
    control_actions_d, descent_rates_d = control_actions[idxs], descent_rates[idxs]
    left_controls_d, right_controls_d = left_controls[idxs], right_controls[idxs]; brake_controls_d = brake_controls[idxs]
    diff_values_d, heading_errors_d, turn_rates_d = diff_values[idxs], heading_errors[idxs], turn_rates[idxs]
    times = np.arange(len(idxs)) * dt * step; fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(221, projection='3d'); ax1.set_xlabel("East (m)"); ax1.set_ylabel("North (m)"); ax1.set_zlabel("Alt (m)"); ax1.set_title('3D Trajectory')
    xmin, xmax = min(xs.min(), xt), max(xs.max(), xt); ymin, ymax = min(ys.min(), yt), max(ys.max(), yt)
    zmin, zmax = 0.0, max(1.0, alt.max()); pad_xy = 0.1 * max(xmax - xmin, ymax - ymin, 1.0)
    ax1.set_xlim(xmin - pad_xy, xmax + pad_xy); ax1.set_ylim(ymin - pad_xy, ymax + pad_xy); ax1.set_zlim(zmin, zmax * 1.1)
    gx = np.linspace(xmin - pad_xy, xmax + pad_xy, 15); gy = np.linspace(ymin - pad_xy, ymax + pad_xy, 15)
    GX, GY = np.meshgrid(gx, gy); GZ = np.zeros_like(GX); ax1.plot_wireframe(GX, GY, GZ, linewidth=0.3, color='gray')
    ax1.plot([xt], [yt], [0], marker='x', markersize=10, linestyle='None', color='red')
    theta = np.linspace(0, 2 * np.pi, 100)
    pad_x = xt + landing_pad_radius_m * np.cos(theta); pad_y = yt + landing_pad_radius_m * np.sin(theta)
    ax1.plot(pad_x, pad_y, 0, color='green', linestyle='--', linewidth=1.5)
    loiter_x = xt + loiter_radius_m * np.cos(theta); loiter_y = yt + loiter_radius_m * np.sin(theta)
    ax1.plot(loiter_x, loiter_y, 0, color='orange', linestyle=':', linewidth=1.5)
    traj_line, = ax1.plot([], [], [], lw=1.5, color='blue')
    s = box_size_m / 2.0; cube_verts = np.array([[-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],[-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s]])
    edges = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
    cube_lines = [ax1.plot([], [], [], lw=1.2, color='black')[0] for _ in range(len(edges))]; info_t = ax1.text2D(0.02, 0.95, "", transform=ax1.transAxes, fontsize=9)
    control_t = ax1.text2D(0.02, 0.90, "", transform=ax1.transAxes, fontsize=9); dist_t = ax1.text2D(0.02, 0.85, "", transform=ax1.transAxes, fontsize=9, color='green', weight='bold')
    heading_t = ax1.text2D(0.02, 0.80, "", transform=ax1.transAxes, fontsize=9); turn_t = ax1.text2D(0.02, 0.75, "", transform=ax1.transAxes, fontsize=9)
    ax2 = fig.add_subplot(222); ax2.set_xlabel("East (m)"); ax2.set_ylabel("North (m)"); ax2.set_title('Top-Down View')
    ax2.set_xlim(xmin - pad_xy, xmax + pad_xy); ax2.set_ylim(ymin - pad_xy, ymax + pad_xy); ax2.grid(True); ax2.axis('equal'); ax2.scatter(xt, yt, color='r', s=100, label='Target')
    ax2.plot(pad_x, pad_y, color='green', linestyle='--', label=f'{landing_pad_radius_m}m Landing Pad')
    ax2.plot(loiter_x, loiter_y, color='orange', linestyle=':', label=f'{loiter_radius_m}m Loiter Radius')
    trajectory_line_2d, = ax2.plot([], [], 'b-', alpha=0.7, label='Trajectory'); current_pos_2d, = ax2.plot([], [], 'ro', markersize=8)
    ax3 = fig.add_subplot(223); ax3.set_xlim(0, times[-1]); ax3.set_ylim(0, zmax*1.1); ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)', color='b'); ax3.set_title('Altitude and Descent Rate'); ax3.grid(True)
    ax4 = ax3.twinx(); ax4.set_ylim(0, max(descent_rates_d.max(), 3.0) + 1); ax4.set_ylabel('Descent Rate (m/s)', color='r')
    altitude_line, = ax3.plot([], [], 'b-', label='Altitude'); descent_rate_line, = ax4.plot([], [], 'r-', label='Descent Rate')
    ax5 = fig.add_subplot(224); ax5.set_xlim(0, times[-1]); ax5.set_ylim(-0.1, 1.1); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Control Position'); ax5.set_title('Control Inputs')
    ax5.grid(True); left_servo_line, = ax5.plot([], [], 'g-', label='Left Servo'); right_servo_line, = ax5.plot([], [], 'b-', label='Right Servo')
    brake_servo_line, = ax5.plot([], [], 'r-', label='Brake Servo'); ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    fig.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01));
    def init(): return []
    def update(j):
        traj_line.set_data(xs_d[:j+1], ys_d[:j+1]); traj_line.set_3d_properties(zs_d[:j+1])
        trajectory_line_2d.set_data(xs_d[:j+1], ys_d[:j+1]); current_pos_2d.set_data([xs_d[j]], [ys_d[j]])
        altitude_line.set_data(times[:j+1], zs_d[:j+1]); descent_rate_line.set_data(times[:j+1], descent_rates_d[:j+1])
        left_servo_line.set_data(times[:j+1], left_controls_d[:j+1]); right_servo_line.set_data(times[:j+1], right_controls_d[:j+1])
        brake_servo_line.set_data(times[:j+1], brake_controls_d[:j+1]); cx, cy, cz = xs_d[j], ys_d[j], zs_d[j]
        for k, (a, b) in enumerate(edges):
            p1 = cube_verts[a] + np.array([cx, cy, cz]); p2 = cube_verts[b] + np.array([cx, cy, cz])
            cube_lines[k].set_data([p1[0], p2[0]], [p1[1], p2[1]]); cube_lines[k].set_3d_properties([p1[2], p2[2]])
        info_t.set_text(f"Position: x={cx:6.1f} m, y={cy:6.1f} m, z={cz:5.1f} m"); control_t.set_text(f"Control: {control_actions_d[j]}")
        dist_to_center = math.sqrt((cx - xt)**2 + (cy - yt)**2); dist_to_edge = max(0, dist_to_center - landing_pad_radius_m)
        dist_t.set_text(f"Dist to Pad: {dist_to_edge:5.1f} m"); heading_t.set_text(f"Heading: {result['traj_heading'][idxs[j]]:.1f}° (Error: {heading_errors_d[j]:.1f}°)")
        turn_t.set_text(f"Turn Rate: {turn_rates_d[j]:.2f}°/s")
        return []
    ani = FuncAnimation(fig, update, frames=len(xs_d), init_func=init, interval=40, blit=False, repeat=False)
    return ani, fig, result

# ----------------------
# Main
# ----------------------
def calculate_landing_score(distance, pad_radius):
    if distance <= 1.0: return 100, "🎯 Bullseye!"; 
    elif distance <= pad_radius / 2: return 75, "✅ Excellent"
    elif distance <= pad_radius: return 50, "👍 Good (On the Pad)" 
    else: return 0, "❌ Missed"
if __name__ == "__main__":
    initial_position = [-7.276595, 112.794020]; target_position  = [-7.275000, 112.794020]
    initial_altitude = 25.0; landing_pad_radius = 5.0
    loiter_radius = 25.0
    
    print(f"Starting simulation from {initial_altitude} m altitude with Circular loiter logic...")
    print(f"Targeting a landing pad with a {landing_pad_radius} m radius.")
    
    animation, fig, result = simulate_and_animate_3d(
        start_lat=initial_position[0], start_lon=initial_position[1],
        target_lat=target_position[0], target_lon=target_position[1],
        wind_speed_ms=8.0, wind_from_deg=180.0,
        sim_time_s=360.0, dt=0.1, max_frames=500, init_alt_m=initial_altitude,
        landing_pad_radius_m=landing_pad_radius,
        loiter_radius_m=loiter_radius,
        landing_alt_m=15.0
    )
    if fig:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]); plt.show()
    if result and len(result["traj_lat"]) > 0:
        final_lat, final_lon = result["traj_lat"][-1], result["traj_lon"][-1]
        start_lat, start_lon = result["traj_lat"][0], result["traj_lon"][0]
        final_x, final_y = _to_local_xy(start_lat, start_lon, final_lat, final_lon)
        target_x, target_y = _to_local_xy(start_lat, start_lon, target_position[0], target_position[1])
        final_distance = math.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        score, rank = calculate_landing_score(final_distance, landing_pad_radius)
        print("\n--- Landing Results ---"); print(f"Final distance from target center: {final_distance:.2f} meters")
        print(f"Landing Rank: {rank}"); print(f"Score: {score}/100"); print("-----------------------")

