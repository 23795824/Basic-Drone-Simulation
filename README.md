# Basic-Drone-Simulation

This is a basic drone simulation using a PID controller. It features a 3D animation of the flight path and accompanying plots of altitude, thrust, and tracking errors to show how the controller moves the drone under gravity and drag.

## Prerequisites

- Python 3.6 or newer  
- Git (to clone the repository)

## Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/23795824/Basic-Drone-Simulation.git
   cd Basic-Drone-Simulation

2. Create a virtual environment

    It’s best practice to isolate project dependencies in a .venv

    - macOS/Linux:

        ```bash

        python3 -m venv .venv
        source .venv/bin/activate

        ```
    
    - Windows (PowerShell):

        ```powershell

            python -m venv .venv
            .\.venv\Scripts\Activate.ps1

        ```

3. Install dependencies

    - Once the virtual environment is active, run:

        ```bash

        pip install --upgrade pip
        pip install -r requirements.txt

        ```

## Running the Simulation

1. From within the activated virtual environment, launch the main script. For example, if your entry point is drone_sim.py, run:

        bash python3 drone_sim.py
        
2. Feel free to modify the design and script to suit your needs.

---
