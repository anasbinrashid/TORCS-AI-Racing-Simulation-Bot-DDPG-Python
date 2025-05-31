# TORCS AI Racing

A comprehensive AI racing project for the TORCS (The Open Racing Car Simulator) environment, utilizing Deep Deterministic Policy Gradient (DDPG) reinforcement learning to train an autonomous racing agent. The project includes a robust learning pipeline, telemetry logging, data analysis tools, and visualization utilities to monitor and evaluate the agent's performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Agent](#training-the-agent)
  - [Analyzing Performance](#analyzing-performance)
  - [Visualizing Results](#visualizing-results)
- [Directory Structure](#directory-structure)
- [Key Components](#key-components)
  - [DDPG Agent](#ddpg-agent)
  - [TORCS Environment](#torcs-environment)
  - [Data Analysis](#data-analysis)
  - [Telemetry Logging](#telemetry-logging)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
TORCS-AI-Racing is designed to train an autonomous racing agent in the TORCS simulator using the DDPG algorithm, a model-free, off-policy actor-critic method suitable for continuous action spaces. The agent learns to control steering, acceleration, and braking to optimize lap times, minimize damage, and maintain track position. The project includes tools for logging telemetry data, analyzing performance metrics, and visualizing training progress and racing behavior.

The codebase is modular, with separate components for the reinforcement learning agent, environment interaction, telemetry logging, and data analysis. It supports both training and evaluation modes, making it suitable for research, experimentation, and deployment in simulated racing scenarios.

## Features
- **DDPG-based Reinforcement Learning**: Implements a DDPG agent with actor and critic networks for continuous control.
- **TORCS Environment Integration**: Custom environment wrapper (`TorcsEnv`) for seamless interaction with the TORCS simulator.
- **Telemetry Logging**: Logs detailed telemetry data (e.g., speed, track position, steering) to CSV files for analysis.
- **Data Analysis Tools**: Provides utilities to analyze lap times, racing lines, and model behavior, with visualization support.
- **Visualization**: Generates plots for training progress, lap times, racing lines, and behavioral analysis (e.g., speed vs. curvature).
- **Recovery Mechanisms**: Includes logic for detecting and recovering from stuck situations or off-track scenarios.
- **Modular Design**: Organized codebase with separate modules for agent, environment, and analysis.
- **Customizable Configuration**: Supports various TORCS configurations (e.g., vision mode, throttle control, gear changes).

## Requirements
- **Operating System**: Linux (Ubuntu recommended) or macOS
- **TORCS**: The Open Racing Car Simulator (version with UDP support)
- **Python**: 3.8+
- **Dependencies**:
  - `numpy`
  - `tensorflow` (2.x)
  - `torch`
  - `pandas`
  - `matplotlib`
  - `gym`
  - `psutil`
  - `subprocess`
  - `xml.etree.ElementTree`
- **Hardware**:
  - CPU: Multi-core processor
  - GPU: Optional (for faster training)
  - RAM: 8GB+
  - Disk Space: 2GB+ (for TORCS and telemetry logs)

## Installation
1. **Install TORCS**:
   - Follow the official TORCS installation guide for your operating system: [TORCS Installation](http://torcs.sourceforge.net/index.php?name=Sections&op=viewarticle&artid=3).
   - Ensure the `scr_server` module is enabled for UDP communication.

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/reponame.git
   cd TORCS-AI-Racing
   ```

3. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with the following:
   ```text
   numpy
   tensorflow
   torch
   pandas
   matplotlib
   gym
   psutil
   ```

5. **Configure TORCS**:
   - Ensure TORCS is configured to run without fuel or lap time limits (handled by `torcs_env.py`).
   - Verify the race configuration file (`raceconfigs/default.xml`) is accessible or create custom configurations.

## Usage

### Training the Agent
1. **Launch TORCS**:
   - Start TORCS in the background with the appropriate configuration:
     ```bash
     torcs -nofuel -nolaptime &
     ```

2. **Run the Training Script**:
   - Use `ddpg.py` to train the DDPG agent:
     ```bash
     python ddpg.py
     ```
   - The script initializes the TORCS environment, trains the agent for a specified number of episodes, and saves model weights periodically (`actormodel.h5`, `criticmodel.h5`).

3. **Monitor Training**:
   - Training progress is logged to the console, including episode rewards and replay buffer size.
   - Telemetry data is saved to `logs/telemetry_<timestamp>.csv`.

### Analyzing Performance
1. **Run the Data Analyzer**:
   - Use `dataAnalyzer.py` to analyze telemetry and training results:
     ```bash
     python dataAnalyzer.py
     ```
   - This script loads the latest telemetry and results files, generates visualizations, and produces a performance report.

2. **Output**:
   - Visualizations are saved in `visualizations/` (e.g., training progress, lap times, racing lines).
   - A summary report is saved in `visualizations/performance_report_<timestamp>.txt`.

### Visualizing Results
- **Training Progress**: Plots rewards and losses over episodes (`visualize_training_progress`).
- **Lap Times**: Bar plots of lap times with the best lap highlighted (`visualize_lap_times`).
- **Racing Line**: Plots track position and speed profile for a specific lap (`visualize_racing_line`).
- **Model Behavior**: Scatter plots analyzing speed vs. curvature, steering behavior, and obstacle avoidance (`analyze_model_behavior`).

## Directory Structure
```
TORCS-AI-Racing/
├── ddpg.py                # Main script for training the DDPG agent
├── tempo.py               # Simplified script for testing TORCS interaction
├── dataAnalyzer.py        # Data analysis and visualization tool
├── driver.py              # Driver class for managing TORCS interaction
├── learningAgent.py       # DDPG agent implementation with actor-critic networks
├── torcs_env.py           # Custom TORCS environment wrapper
├── OU.py                  # Ornstein-Uhlenbeck process for exploration noise
├── ReplayBuffer.py        # Experience replay buffer implementation
├── Launcher1.py           # Client-server communication for TORCS
├── logs/                  # Directory for telemetry logs
├── models/                # Directory for saved model weights
├── results/               # Directory for training results
├── visualizations/        # Directory for analysis visualizations
├── raceconfigs/           # Directory for TORCS race configuration files
└── README.md              # Project documentation
```

## Key Components

### DDPG Agent
- **File**: `learningAgent.py`, `ddpg.py`
- **Description**: Implements a DDPG agent with actor and critic networks using PyTorch and TensorFlow. The agent learns to map states (e.g., speed, track position, sensors) to actions (steering, acceleration, brake).
- **Key Features**:
  - Continuous action space handling
  - Exploration noise via Ornstein-Uhlenbeck process
  - Experience replay buffer
  - Soft target network updates
  - Reward function balancing speed, track position, and damage

### TORCS Environment
- **File**: `torcs_env.py`
- **Description**: A Gym-compatible wrapper for the TORCS simulator, handling state observation, action execution, and reward calculation.
- **Key Features**:
  - Supports throttle and gear change options
  - Configurable race settings (e.g., vision mode, damage)
  - Random track initialization for varied training
  - Episode termination conditions (e.g., off-track, low progress)

### Data Analysis
- **File**: `dataAnalyzer.py`
- **Description**: Analyzes telemetry data and training results to provide insights into agent performance.
- **Key Features**:
  - Loads telemetry from CSV files and results from pickle files
  - Generates visualizations for training progress, lap times, and racing lines
  - Analyzes model behavior (e.g., speed vs. curvature, steering vs. track position)
  - Produces a detailed performance report with metrics like average reward, lap times, and damage

### Telemetry Logging
- **File**: `driver.py`
- **Description**: Logs detailed telemetry data during simulation, including car state, control inputs, and rewards.
- **Key Features**:
  - Saves data to CSV files in `logs/`
  - Includes timestamped records for analysis
  - Supports logging of track sensors, opponent distances, and wheel velocities

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **TORCS Community**: For providing the open-source racing simulator.
- **OpenAI Gym**: For the environment interface inspiration.
- **PyTorch & TensorFlow**: For deep learning frameworks.
- **Research Papers**:
  - Lillicrap et al., "Continuous control with deep reinforcement learning" (DDPG paper)
