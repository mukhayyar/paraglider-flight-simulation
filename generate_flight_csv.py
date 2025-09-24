import math
import numpy as np
import csv

# ----------------------
# Utility functions (Needed for sim)
# ----------------------
R_EARTH = 6371000.0
def wrap_deg(a: float) -> float:
    while a > 180: a -= 360
    while a < -180: a += 360
    return a

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

def step_latlon(lat, lon, vx, vy, dt):
    dN = vy*dt; dE = vx*dt
    dlat = (dN/R_EARTH) * 180/math.pi
    dlon = (dE/(R_EARTH*math.cos(math.radians(lat)))) * 180/math.pi
    return lat + dlat, lon + dlon

# ----------------------
# Simulation core (Same as before)
# ----------------------
class ParagliderSim:
    # (Copy the entire ParagliderSim class from our previous code here)
    # ...
    # (I'm omitting the full class here for brevity, 
    # but you must copy/paste it in this file)
    def __init__(self,
                 start_lat, start_lon,
                 target_lat, target_lon,
                 wind_speed_ms=5.0, wind_from_deg=270.0,
                 v_trim_ms=10.0,
                 sink_trim_ms=1.2, sink_max_ms=2.5,
                 kp_track=0.05, max_diff=0.8, track_blend=1.0,
                 approach_radius_m=60.0, approach_alt_m=40.0, flare_alt_m=5.0,
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
        self.flare_alt = float(flare_alt_m); self.INIT, self.CRUISE, self.APPROACH, self.FLARE, self.DISARM = range(5)
        self.state = self.CRUISE; self.left = 0.5; self.right = 0.5; self.brake = 0.25
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
        e_track = wrap_deg(brg - cog); e_head  = wrap_deg(brg - self.heading)
        e = (self.track_blend * e_track) + ((1.0 - self.track_blend) * e_head)
        if self.state == self.CRUISE and (d < self.approach_radius or self.alt < self.approach_alt): self.state = self.APPROACH
        if self.state == self.APPROACH and self.alt <= self.flare_alt and d < 30: self.state = self.FLARE
        if self.state == self.FLARE and self.alt <= 0.1: self.state = self.DISARM
        control_action = ""; e_corrected = 0.0
        if abs(e) > self.heading_deadband: e_corrected = e - np.sign(e) * self.heading_deadband
        diff = max(-self.max_diff, min(self.max_diff, self.kp * e_corrected))
        if self.alt > self.min_control_alt:
            if self.state in (self.CRUISE, self.APPROACH):
                self.left  = 0.5 - 0.5 * diff; self.right = 0.5 + 0.5 * diff
                if self.state == self.CRUISE: self.brake = 0.25; control_action = "Cruising"
                else:
                    base_brake = 0.30
                    current_sink = self.sink_trim + (self.sink_max - self.sink_trim) * max(0.0, (self.brake - 0.2) / 0.8)
                    error = self.target_descent_rate - current_sink
                    self.descent_integral += error * self.dt; derivative = (error - self.last_descent_error) / self.dt
                    self.last_descent_error = error
                    pid_output = (self.KP_DESCENT * error + self.KI_DESCENT * self.descent_integral + self.KD_DESCENT * derivative)
                    self.brake = np.clip(base_brake + pid_output, 0.0, 1.0)
                    control_action = "Braking/Turning"
            elif self.state == self.FLARE:
                self.brake = min(1.0, self.brake + 0.02); flare_gain = 0.3
                diff_flare = max(-self.max_diff * flare_gain, min(self.max_diff * flare_gain, self.kp * e_corrected * flare_gain))
                self.left  = 0.5 - 0.5 * diff_flare; self.right = 0.5 + 0.5 * diff_flare
                control_action = "Flare (Landing)"
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
        self.traj_state.append(self.state); self.traj_heading.append(self.heading)
        # ... (rest of logging)
        
    def run(self):
        steps = int(self.sim_time_s / self.dt)
        for i in range(steps):
            if self.state == self.DISARM and self.alt <= 0.1: break
            self.step()
        return {k: np.array(v) for k, v in self.__dict__.items() if k.startswith("traj_")}

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    
    # --- Simulation Parameters ---
    initial_position = [-7.276595, 112.794020]
    target_position  = [-7.275000, 112.794020]
    initial_altitude = 250.0
    sim_dt = 0.1 # Simulation timestep
    
    print("Running simulation to generate flight data...")
    sim = ParagliderSim(
        start_lat=initial_position[0], 
        start_lon=initial_position[1],
        target_lat=target_position[0], 
        target_lon=target_position[1],
        wind_speed_ms=8.0,
        wind_from_deg=180.0,
        sim_time_s=360.0,
        dt=sim_dt,
        init_alt_m=initial_altitude
    )
    result = sim.run()

    # --- CSV Writing ---
    output_filename = "flight_log.csv"
    
    # We want 1Hz data, and the sim runs at 0.1s steps (10Hz)
    # So, we take every 10th sample.
    sample_rate_hz = 1.0
    sample_step = int((1.0 / sample_rate_hz) / sim_dt) # 1.0 / 0.1 = 10
    
    # Get the data arrays
    lats = result["traj_lat"]
    lons = result["traj_lon"]
    alts = result["traj_alt"]
    headings = result["traj_heading"]
    
    # Define the headers
    headers = ["gps_lat", "gps_lon", "altitude", "heading"]

    try:
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers) # Write the header
            
            # Iterate and sample at 1Hz
            for i in range(0, len(lats), sample_step):
                writer.writerow([
                    lats[i],
                    lons[i],
                    alts[i],
                    headings[i]
                ])
                
        print(f"Successfully generated '{output_filename}' with {len(lats)//sample_step} data points.")

    except Exception as e:
        print(f"Error writing CSV file: {e}")