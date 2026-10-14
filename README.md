# AI Night-Patrol Drone Simulation

A Python-based **Central Drone Monitoring System** that simulates a drone patrolling a fenced compound at night. The drone observes simulated targets and sends behavioural measurements to a separate command-center server, which classifies tracks and manages alerts.

> **Project scope:** This is a simulation and learning project. Its threat model is trained using simulated data, not real-world intrusion data. Its predictions should not be treated as a reliable real-world security system.

## Features

- Graphical night-patrol simulation built with Pygame.
- Simulated thermal-camera view and optional satellite-map background.
- Simulated authorised staff, animals, and intruders.
- Drone movement and patrol logic, including PID control and a simulated wind disturbance.
- Target tracking with a constant-velocity Kalman filter and track-confirmation logic.
- Behaviour-based threat scoring using four features:
  - **Speed** — how quickly a target moves.
  - **Loitering** — how much a target appears to remain in an area.
  - **Proximity** — distance from the protected asset.
  - **Route deviation** — distance from the authorised walkway.
- Separate command-center service with a live browser dashboard, charts, track table, and incident log.
- Asynchronous communication between the simulation and command center, so network requests do not run in the main graphics loop.
- Incident merging to group nearby alerts occurring within a configured time window.

## Architecture

```text
Pygame Drone Simulation
        |
        | behavioural observations over HTTP
        v
Command Center (Python HTTP server)
        |
        v
Threat Engine
        |
        +--> THREAT / SUSPICIOUS / NOT_THREAT verdict
        +--> incident and alert management
        +--> dashboard data and evaluation counters
        |
        v
Browser Dashboard: http://127.0.0.1:8000
```

The simulation sends observations; the command center makes the threat prediction when the default `COMMAND_CENTER` mode is enabled. The simulator's ground-truth label is used for evaluation counters only, not as a prediction feature.

## Project Structure

```text
project-folder/
├── night_patrol_drone.py   # Pygame simulation and drone GUI
├── patrol_core.py          # Simulation physics, camera model, tracking and training/evaluation logic
├── command_center.py       # HTTP command-center server and dashboard
├── threat_engine.py        # Threat scoring, verdicts and incident management
├── datasets/
│   └── threat_model.json   # Trained model file (generated or loaded at runtime)
├── requirements.txt        # Python dependencies (create as described below)
└── README.md
```

The `datasets/threat_model.json` file may be created automatically when the application trains the model for the first time. Keep the generated file if you want to reuse the same trained model.

## Requirements

- Python 3.10 or newer recommended.
- `numpy`
- `pygame`
- `requests`

The command-center server primarily uses the Python standard library, but model training relies on the simulation core and NumPy.

## Setup

1. Install Python and make sure `python` works in your terminal.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the dependencies:

   ```bash
   python -m pip install numpy pygame requests
   ```

5. If you prefer a dependency file, create `requirements.txt` with:

   ```text
   numpy
   pygame
   requests
   ```

## Run the Project

Open **two terminals** in the project folder.

### Terminal 1 — Start the command center

```bash
python command_center.py
```

The server uses `http://127.0.0.1:8000`. On the first run, it may train the model if `datasets/threat_model.json` is missing.

Open the dashboard in your browser:

**http://127.0.0.1:8000**

### Terminal 2 — Start the drone simulation

```bash
python night_patrol_drone.py
```

The simulation is configured to send observations to the command center at `http://127.0.0.1:8000`.

> Start the command center first so it can receive the simulation's observations and return verdicts.

## Simulation Controls

| Key | Action |
|---|---|
| `SPACE` | Pause or resume the simulation |
| `R` | Reset the simulation |
| `N` | Toggle night-look display |
| `G` | Show or hide simulation ground truth |
| `M` | Toggle satellite background |
| `F` | Change simulation speed (1×, 2×, 4×, 8×) |
| `T` | Spawn an intruder |
| `ESC` | Quit |

## Command Center API

The server exposes these endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /api/observations` | Receive a drone observation and return a verdict |
| `POST /api/world-state` | Receive optional simulation counters and target state |
| `POST /api/session/reset` | Reset the current session state |
| `POST /api/session/stop` | Mark the session as stopped |
| `GET /api/state` | Return dashboard/session data |
| `GET /api/model` | Return model information |

## How Threat Scoring Works

The threat engine standardises the behavioural features and applies a logistic-regression score to estimate a threat probability. The probability is smoothed over successive observations for each track.

The configured thresholds are:

- **THREAT:** probability at or above `0.5`
- **SUSPICIOUS:** probability at or above `0.3` but below `0.5`
- **NOT_THREAT:** probability below `0.3`
- **UNVERIFIED:** behavioural features are not yet available for the track

These thresholds and model parameters are implementation settings, not proof of real-world accuracy. The dashboard's TP, TN, FP, and FN counters compare predictions with simulated ground truth for demonstration and evaluation.

## Limitations and Responsible Use

- The world, people, sensor measurements, and training data are simulated.
- The thermal display is a synthetic visualisation, not a live thermal-camera feed.
- Satellite tiles may require an internet connection and may be cached locally.
- The default map coordinates are placeholders; change them in `night_patrol_drone.py` if you intentionally want a different background location.
- This project does not establish a person's intent or reliably detect real intrusions. Do not use it as the sole basis for real security decisions.

## Troubleshooting

- **Dashboard does not open:** confirm `command_center.py` is running and visit `http://127.0.0.1:8000`.
- **Missing package error:** run `python -m pip install numpy pygame requests`.
- **Model file missing:** allow the first-run training to finish. If it fails, check the terminal output and confirm the project files and `datasets` directory are writable.
- **Simulation shows unverified tracks:** check that the command center is running and that the configured server URL is correct.
- **Port 8000 is already in use:** stop the other process using that port before starting the server.

## Future Improvements

- Add automated tests for tracking, scoring, and API endpoints.
- Store incident history persistently between sessions.
- Add configurable patrol zones and camera parameters.
- Evaluate the model on a separate, documented test dataset.
- Improve dashboard accessibility and mobile responsiveness.


