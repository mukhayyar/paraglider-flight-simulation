
# CANSAT2026 — Paraglider Flight Simulation & Visualization

This repository contains a small set of Python tools used to simulate a paraglider-style flight, export a flight log (CSV), and animate / visualize the trajectory in 2D/3D. The code was developed as part of a CANSAT project (2026) and includes several variants of the simulator and animation helpers.

Key capabilities
- Generate realistic simulated flight data (latitude, longitude, altitude, heading) with configurable wind, trim speed, sink rates and simple autopilot logic.
- Export the simulated trajectory to a CSV file suitable for playback or downstream processing.
- Visualize and animate the flight log in 3D (with landing pad / target markers), including descent and turn rate overlays.

Repository layout
- `generate_flight_csv.py` — Simulation script that runs a ParagliderSim and writes a `flight_log.csv`. Parameters like start/target coords, wind, timestep and simulation duration are defined in the file and can be adjusted.
- `animate_flight_csv.py` — Reads a CSV flight log and animates the trajectory using matplotlib. The animation shows a 3D flight path, landing pad, and additional telemetry plots.
- `simulation_animation_google.py` — An upgraded simulation + animation script that includes a PD loiter controller and more advanced behavior. Useful for experimenting with loitering, approach and flare behaviours.
- `main.py` — (may contain helper wiring or utilities; inspect the file for entry points used in your workflow).
- `flight_log.csv` — Example/previously-generated CSV flight log (if present).
- `requirements.txt`, `pyproject.toml` — Python dependency manifests.

Requirements
- Python 3.9 or newer
- The project depends on common scientific and plotting packages. The included `pyproject.toml` and `requirements.txt` list pinned versions used during development (e.g. numpy, matplotlib). Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

or, if you prefer the project metadata in `pyproject.toml`, use your preferred tool (poetry, pip with build tools, etc.).

Quick start

1) Generate a flight log (CSV)

Open `generate_flight_csv.py` and adjust the simulation parameters near the bottom of the file if needed (start/target coordinates, wind, initial altitude, dt, sim_time_s). Then run:

```bash
python3 generate_flight_csv.py
```

This will run the simulator and produce a `flight_log.csv` (default filename in the script). The script runs the ParagliderSim and writes CSV columns such as `gps_lat`, `gps_lon`, `altitude`, and `heading` at a sampling rate (usually 1 Hz by default; the simulator may run at a higher internal dt).

2) Animate the flight log

Open `animate_flight_csv.py`. The animator expects a CSV with lat/lon/alt/heading columns. The file includes a function `animate_csv_data(csv_filename, target_coords, ...)` — update the `target_coords` or call the function with your desired coordinates. Then run:

```bash
python3 animate_flight_csv.py
```

If you prefer, you can import `animate_csv_data` from a Python REPL or another script and pass the CSV path and target coordinates programmatically.

Notes about configuration
- Most parameters (wind speed, wind direction, v_trim, sink rates, PID gains, approach radius, loiter radius) are defined as named arguments in the simulator constructors near the top of `generate_flight_csv.py` and `simulation_animation_google.py`. Edit those values to tune behavior.
- The animation scripts assume the target/landing pad location is either embedded in the CSV or passed as an argument; check `animate_flight_csv.py` for the variable `target_coords` or similar and set it to the desired lat/lon.
- Some functions in the repository use helper utilities (distance, bearing, lat/lon stepping). If you see "omitted" or placeholder comments in the file, inspect the function bodies — a few areas in the bundled attachments were shortened for clarity.

Development & testing
- There are no unit tests included in the repository by default. If you change simulation logic, add small unit tests for helper math (distance, bearing, wrapping) and a short regression test that runs the simulation for a few seconds and verifies output shape.
- To speed up experimentation, reduce `sim_time_s` and increase `dt`.

Tips
- Use a virtual environment (venv) to isolate dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- If you want to save animation output to a file, modify the matplotlib animation writer settings in `animate_flight_csv.py` (look for `FuncAnimation` usage and `animation.save(...)`). Be aware that saving to video may require extra packages (ffmpeg).

Contributing
- Feel free to open issues or add improvements. Suggested enhancements:
	- Add a simple CLI for the generator/animator so parameters can be passed via command-line flags.
	- Add unit tests that validate core math functions.
	- Add an option to export animations as MP4/GIF using ffmpeg/imageio.

License
- No license file is provided in the repository. Add a LICENSE file (e.g., MIT) if you intend to publish or share this code publicly.

Contact / Next steps
- Inspect the top of each script for configuration blocks and tweak values to match your test scenarios. If you'd like, I can:
	- Add a small CLI wrapper for `generate_flight_csv.py` and `animate_flight_csv.py`.
	- Implement example unit tests and a small CI workflow.

Enjoy exploring the simulation and visualizations — let me know which next step you'd like me to implement.

