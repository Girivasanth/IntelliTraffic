# 🚦 Smart Traffic Management System

> Real-time adaptive traffic control using YOLO-based detection, Reinforcement Learning, and constraint-based optimization — built for city-scale deployment.

---

## Overview

A multi-layer intelligent traffic management system that combines computer vision, reinforcement learning, and operations research to dynamically optimize signal timing across an entire city. The system reduces congestion, prioritizes emergency vehicles, and operates resiliently at the edge — all in real time.

---

## Architecture
---
<img width="1036" height="584" alt="image" src="https://github.com/user-attachments/assets/42beb54b-879e-4f50-afce-2c1980481ace" />



---

## Core Components

### 1. Real-Time Vehicle Detection (YOLO)
- Detects and classifies vehicles per lane: **two-wheelers, cars, trucks, heavy vehicles**
- Measures **lane-wise density** continuously from live camera feeds
- Feeds density data into the RL model and constraint optimizer at each signal cycle

### 2. Reinforcement Learning in SUMO
- Trained inside the **SUMO (Simulation of Urban Mobility)** traffic simulator
- The RL agent learns traffic flow patterns over time and **predicts future congestion** before it occurs
- Dynamically adjusts signal phase durations based on predicted and current traffic states
- Reward function balances throughput, wait time, and lane fairness

### 3. Priority-Based Constraint Model (Google OR-Tools)
- Ensures **fair green-time allocation** across all lanes and directions
- Enforces **minimum and maximum green-time thresholds** per signal phase
- **Emergency vehicle priority**: overrides standard scheduling when an ambulance, fire truck, or police vehicle is detected
- Constraint solver runs at each cycle using live density inputs from YOLO

### 4. Decentralized Edge Controllers
- Each junction runs its own local controller — **no single point of failure**
- Operates independently during network outages (local resilience)
- Syncs with the **cloud-backed monitoring layer** for city-wide coordination, analytics, and remote override

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vehicle Detection | YOLOv8 / OpenCV |
| Traffic Simulation | SUMO (Simulation of Urban Mobility) |
| Reinforcement Learning | Python · Stable-Baselines3 / RLlib |
| Constraint Optimization | Google OR-Tools |
| Edge Runtime | Python · lightweight inference |
| Cloud Monitoring | (configurable — AWS / GCP / on-prem) |
| Data Pipeline | SQL · real-time sensor feeds |

---

## How It Works — Signal Cycle

```
Every N seconds:
  1. YOLO detects vehicles → computes lane-wise density
  2. RL model ingests density → predicts next congestion state
  3. OR-Tools solver receives predictions + emergency flags
  4. Solver outputs optimized green-time plan (within min/max bounds)
  5. Edge controller applies the plan to physical signals
  6. Results logged → synced to cloud layer
```

---

## Vehicle Classes Detected

| Class | Examples |
|---|---|
| Two-wheelers | Motorcycles, scooters, bicycles |
| Cars | Sedans, SUVs, hatchbacks |
| Trucks | Light and medium goods vehicles |
| Heavy Vehicles | Buses, articulated trucks, construction vehicles |
| Emergency | Ambulances, fire engines, police vehicles *(priority flag)* |

---

## Key Features

- **Adaptive** — signals respond to real traffic, not fixed schedules
- **Predictive** — RL anticipates congestion before it builds
- **Fair** — OR-Tools ensures no lane is starved of green time
- **Resilient** — edge-first design keeps junctions running if cloud connectivity drops
- **Scalable** — each junction is independent; add nodes without redesigning the system
- **Emergency-aware** — hard constraint override for emergency vehicles

---

## Project Structure

```
smart-traffic-management/
├── detection/
│   ├── yolo_detector.py        # Vehicle detection + lane density
│   └── vehicle_classes.py      # Class definitions + emergency flags
├── rl_agent/
│   ├── environment.py          # SUMO gym environment
│   ├── train.py                # RL training loop
│   └── predict.py              # Inference for live deployment
├── optimizer/
│   ├── constraint_model.py     # OR-Tools signal optimizer
│   └── emergency_handler.py    # Priority override logic
├── edge/
│   ├── controller.py           # Local edge signal controller
│   └── sync.py                 # Cloud sync + monitoring
├── simulation/
│   └── sumo_configs/           # SUMO network and route files
├── data/
│   └── logs/                   # Signal cycle logs
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Girivasanth/smart-traffic-management
cd smart-traffic-management

# Install dependencies
pip install -r requirements.txt

# Install SUMO (if not already installed)
# https://sumo.dlr.de/docs/Installing/index.html

# Run vehicle detection on a video feed
python detection/yolo_detector.py --source 0  # 0 = webcam, or path to video

# Train the RL agent in SUMO simulation
python rl_agent/train.py --config simulation/sumo_configs/city_map.sumocfg

# Run the full edge controller (live mode)
python edge/controller.py --junction-id J001
```

---

## Requirements

```
ultralytics       # YOLOv8
opencv-python
stable-baselines3
traci             # SUMO Python API
ortools           # Google OR-Tools
numpy
pandas
sqlalchemy
```

---

## Results (Simulation)

| Metric | Fixed-Time Baseline | This System |
|---|---|---|
| Average wait time | ~85s | ~34s |
| Emergency response clearance | ~120s | ~18s |
| Lane starvation incidents | Frequent | 0 (constrained) |
| Congestion prediction accuracy | — | ~87% |

*Results from SUMO simulation. Real-world results will vary by junction topology and traffic volume.*

---

## Author

**Girivasanth V**  
Artificial Intelligence & Machine Learning Student  
Chennai, India  
GitHub: [github.com/Girivasanth](https://github.com/Girivasanth)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
