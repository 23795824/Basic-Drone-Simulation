import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import matplotlib.gridspec as gridspec

# -------------------------------------
# Improved PID Controller
# -------------------------------------
# We chose a PID Controller for this design because it provides
# a good balance between responsiveness and stability for position control.
# Variants like LQR or MPC could be used, but PID is simpler to tune
# and understand for this demonstration.
class PIDController:
    def __init__(self, Kp, Ki, Kd, anti_windup_limit=10.0, filter_coeff=0.1):
        # Proportional, Integral, and Derivative gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # Store previous error for derivative calculation
        self.prev_error = 0.0
        
        # Integral accumulator
        self.integral = 0.0
        
        # Smoothed derivative (low-pass filter state)
        self.filtered_derivative = 0.0
        
        # Anti-windup limit prevents the integral term from growing unbounded
        self.anti_windup_limit = anti_windup_limit
        
        # Coefficient for exponential moving average on derivative
        # A small filter_coeff means more smoothing (less noise) at the cost of slower response
        self.filter_coeff = filter_coeff

    def compute(self, setpoint, current, dt):
        # Avoid divide-by-zero when computing derivative
        if dt <= 0:
            return 0.0
        
        # Compute current error between desired setpoint and actual measurement
        error = setpoint - current
        
        # Anti-windup: only accumulate integral if it is within bounds
        # This prevents excessive overshoot when the controller is saturated
        if abs(self.integral) < self.anti_windup_limit:
            self.integral += error * dt
        
        # Compute raw derivative (change in error over time)
        raw_derivative = (error - self.prev_error) / dt
        
        # Apply a simple first-order low-pass filter to the derivative term
        # This reduces sensitivity to high-frequency noise in the error signal
        self.filtered_derivative = self.filter_coeff * raw_derivative + \
                                  (1 - self.filter_coeff) * self.filtered_derivative
        
        # PID output: P term + I term + D term
        output = self.Kp * error + \
                 self.Ki * self.integral + \
                 self.Kd * self.filtered_derivative
        
        # Store error for next derivative calculation
        self.prev_error = error
        
        return output

# -------------------------------------
# Enhanced 3D Drone Model 
# (state = [x, y, z, vx, vy, vz])
# -------------------------------------
# We include both linear and quadratic drag to make the dynamics more realistic.
# At higher speeds, the quadratic drag term becomes significant.
class Drone3D:
    def __init__(self, mass=1.0, gravity=9.81, linear_drag=0.1, quadratic_drag=0.01):
        self.mass = mass         # Drone mass (kg)
        self.gravity = gravity   # Gravitational acceleration (m/s²)
        
        # Drag coefficients: linear drag approximates frictional forces at low speed,
        # quadratic drag approximates aerodynamic drag that grows with velocity squared.
        self.linear_drag = linear_drag     
        self.quadratic_drag = quadratic_drag  
        
        # Initial state: x, y, z positions (m) and vx, vy, vz velocities (m/s)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    def dynamics(self, state, t, thrust):
        """
        Enhanced dynamics model with realistic drag
        
        state: [x, y, z, vx, vy, vz]
        thrust: tuple (Tx, Ty, Tz) control forces in each axis
        
        We return the time derivatives: [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        Tx, Ty, Tz = thrust
        
        # Compute drag forces in each axis. Quadratic component uses |v|*v 
        # so that drag always opposes motion.
        drag_x = self.linear_drag * vx + self.quadratic_drag * np.abs(vx) * vx
        drag_y = self.linear_drag * vy + self.quadratic_drag * np.abs(vy) * vy
        drag_z = self.linear_drag * vz + self.quadratic_drag * np.abs(vz) * vz
        
        # Compute accelerations by Newton's 2nd law: (thrust - drag - weight) / mass
        ax = (Tx - drag_x) / self.mass
        ay = (Ty - self.mass * self.gravity - drag_y) / self.mass
        az = (Tz - drag_z) / self.mass
        
        # Return derivative of state
        return [vx, vy, vz, ax, ay, az]

# -------------------------------------
# Setpoint Functions for a Flight Cycle.
# -------------------------------------
# Define a smooth flight profile: takeoff, cruise (horizontal move), then landing.
# The interior of each phase uses a continuous, differentiable function to avoid
# large jerks in acceleration.
def get_cycle_setpoints(t_cycle, x0, z0, alt_target, x_target, z_target):
    if t_cycle < 3.0:
        # TAKEOFF: Use a cosine-based ramp to smoothly increase altitude from 0 to alt_target
        # This avoids a sudden step change in setpoint at t=0.
        sp_y = alt_target * (0.5 - 0.5 * np.cos(np.pi * t_cycle / 3.0))
        sp_x = x0  # Keep horizontal position fixed during takeoff
        sp_z = z0
    elif t_cycle < 7.0:
        # CRUISE: Maintain altitude at alt_target; move horizontally from (x0, z0) to (x_target, z_target).
        sp_y = alt_target
        # Normalize the cruise phase time to [0,1]
        tau = (t_cycle - 3.0) / 4.0  
        # Third-order polynomial (S-curve) ensures zero initial and final velocity 
        # for smooth acceleration/deceleration.
        s = 3 * tau**2 - 2 * tau**3
        sp_x = x0 + (x_target - x0) * s
        sp_z = z0 + (z_target - z0) * s
    else:
        # LANDING: Reverse of takeoff: smoothly descend from alt_target to 0
        sp_y = alt_target * (0.5 + 0.5 * np.cos(np.pi * (t_cycle - 7.0) / 3.0))
        sp_x = x_target  # Stay at landing horizontal coordinates
        sp_z = z_target
    return sp_x, sp_y, sp_z

# -------------------------------------
# Simulation for Multiple Flight Cycles with improved controllers
# -------------------------------------
def simulate_multi_cycle(num_cycles=3, T_cycle=10.0, dt=0.05):
    """
    Run multiple sequential flight cycles (takeoff → cruise → landing).
    - num_cycles: number of independent flight cycles
    - T_cycle: duration of each cycle (seconds)
    - dt: simulation timestep (seconds)
    
    We store time history, positions, setpoints, thrusts, and tracking errors
    for later analysis and animation.
    """
    steps_per_cycle = int(T_cycle / dt)
    total_steps = num_cycles * steps_per_cycle

    # Preallocate arrays for speed and memory efficiency
    history = {
        'time': np.zeros(total_steps),
        'x': np.zeros(total_steps),
        'y': np.zeros(total_steps),
        'z': np.zeros(total_steps),
        'setpoint_x': np.zeros(total_steps),
        'setpoint_y': np.zeros(total_steps),
        'setpoint_z': np.zeros(total_steps),
        'thrust_x': np.zeros(total_steps),
        'thrust_y': np.zeros(total_steps),
        'thrust_z': np.zeros(total_steps),
        'error_x': np.zeros(total_steps),
        'error_y': np.zeros(total_steps),
        'error_z': np.zeros(total_steps)
    }

    # Initialize the drone with default physical parameters
    drone = Drone3D()
    
    # Create separate PID controllers for each axis with tuned gains.
    # We set Ki relatively small to avoid overshoot, and Kd moderate for damping.
    pid_x = PIDController(Kp=2.0, Ki=0.2, Kd=1.5, anti_windup_limit=5.0)
    pid_y = PIDController(Kp=4.0, Ki=0.5, Kd=2.0, anti_windup_limit=5.0)
    pid_z = PIDController(Kp=2.0, Ki=0.2, Kd=1.5, anti_windup_limit=5.0)

    step_index = 0
    time_accum = 0.0

    for cycle in range(num_cycles):
        # For cycle 0, ensure drone starts on the ground (y=0). For subsequent cycles,
        # the state carries over (simulate multiple landings/takeoffs back to back).
        if cycle == 0:
            drone.state[1] = 0.0  # y-position (altitude) = 0
        
        # Record initial horizontal coords for setpoint functions
        x0 = drone.state[0]
        z0 = drone.state[2]
        
        # Randomly choose a new landing site within ±3 m in x and z
        # This illustrates robustness of controller to different target locations
        x_target = x0 + np.random.uniform(-3, 3)
        z_target = z0 + np.random.uniform(-3, 3)
        
        # Choose a random cruise altitude between 4 and 6 m each cycle
        alt_target = np.random.uniform(4, 6)

        for j in range(steps_per_cycle):
            t_cycle = j * dt
            history['time'][step_index] = time_accum + t_cycle

            # Compute the desired setpoints at this moment in the cycle
            sp_x, sp_y, sp_z = get_cycle_setpoints(t_cycle, x0, z0, alt_target, x_target, z_target)
            history['setpoint_x'][step_index] = sp_x
            history['setpoint_y'][step_index] = sp_y
            history['setpoint_z'][step_index] = sp_z
            
            # Compute tracking errors for analysis/plotting
            error_x = sp_x - drone.state[0]
            error_y = sp_y - drone.state[1]
            error_z = sp_z - drone.state[2]
            history['error_x'][step_index] = error_x
            history['error_y'][step_index] = error_y
            history['error_z'][step_index] = error_z

            # Compute PID commands for each axis
            control_x = pid_x.compute(sp_x, drone.state[0], dt)
            control_y = pid_y.compute(sp_y, drone.state[1], dt)
            control_z = pid_z.compute(sp_z, drone.state[2], dt)
            
            # Feed-forward term in Y-axis: counteract gravity so that PID only
            # needs to handle altitude tracking. This improves stability.
            ff_y = drone.mass * drone.gravity

            # Add velocity-based damping for additional stability (approximate anti-windup
            # on derivative or add additional damping). Here we use a simple -1*v term.
            damping_x = -1.0 * drone.state[3]  # approximate derivative control for X
            damping_y = -1.0 * drone.state[4]  # approximate derivative control for Y
            damping_z = -1.0 * drone.state[5]  # approximate derivative control for Z

            # Apply control limits (simulate actuator saturation):
            #   X and Z thrusters can push ±20 N,
            #   Y thruster (vertical) can only push upward (0 to 30 N) because
            #   negative thrust would not cause downward acceleration beyond gravity.
            thrust_x = np.clip(control_x + damping_x, -20, 20)
            thrust_y = np.clip(ff_y + control_y + damping_y, 0, 30)
            thrust_z = np.clip(control_z + damping_z, -20, 20)

            history['thrust_x'][step_index] = thrust_x
            history['thrust_y'][step_index] = thrust_y
            history['thrust_z'][step_index] = thrust_z

            thrust = (thrust_x, thrust_y, thrust_z)
            t_span = [0, dt]
            
            # Use a numerical integrator (ODE solver) to propagate the drone's state.
            # This captures the continuous-time dynamics more accurately than simple Euler.
            sol = odeint(drone.dynamics, drone.state, t_span, args=(thrust,))
            drone.state = sol[-1]  # Take the state at the end of the time step

            # Log new state into history
            history['x'][step_index] = drone.state[0]
            history['y'][step_index] = drone.state[1]
            history['z'][step_index] = drone.state[2]
            step_index += 1
            
        time_accum += T_cycle  # Move to next cycle's absolute time

    return history, dt

# -------------------------------------
# Enhanced Animation for Time-Series (Altitude, Thrust, and Errors)
# -------------------------------------
def animate_time_plots(history, dt):
    """
    Create a multi-panel animation showing:
      1) Altitude vs. Time (actual vs. setpoint)
      2) Thrust commands in X, Y, Z vs. Time
      3) Tracking errors in X, Y, Z vs. Time
    
    We use GridSpec to allocate relative height ratios for each subplot.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])
    
    # 1) Altitude Plot
    ax1 = plt.subplot(gs[0])
    ax1.set_xlim(history['time'][0], history['time'][-1])
    ax1.set_ylim(0, np.max(history['y']) + 1)
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Drone Flight: Altitude')
    altitude_line, = ax1.plot([], [], 'b-', label='Actual Altitude')
    setpoint_line, = ax1.plot([], [], 'r--', label='Target Altitude')
    ax1.legend()
    ax1.grid(True)

    # 2) Thrust Plot
    ax2 = plt.subplot(gs[1])
    ax2.set_xlim(history['time'][0], history['time'][-1])
    thrust_min = min(np.min(history['thrust_x']),
                     np.min(history['thrust_y']),
                     np.min(history['thrust_z']))
    thrust_max = max(np.max(history['thrust_x']),
                     np.max(history['thrust_y']),
                     np.max(history['thrust_z']))
    ax2.set_ylim(thrust_min - 1, thrust_max + 1)
    ax2.set_ylabel('Thrust (N)')
    ax2.set_title('Control Forces')
    thrust_line_x, = ax2.plot([], [], 'r-', label='Thrust X')
    thrust_line_y, = ax2.plot([], [], 'g-', label='Thrust Y')
    thrust_line_z, = ax2.plot([], [], 'm-', label='Thrust Z')
    ax2.legend()
    ax2.grid(True)
    
    # 3) Tracking Error Plot
    ax3 = plt.subplot(gs[2])
    ax3.set_xlim(history['time'][0], history['time'][-1])
    error_min = min(np.min(history['error_x']),
                     np.min(history['error_y']),
                     np.min(history['error_z']))
    error_max = max(np.max(history['error_x']),
                     np.max(history['error_y']),
                     np.max(history['error_z']))
    margin = max(abs(error_min), abs(error_max)) * 0.1
    ax3.set_ylim(error_min - margin, error_max + margin)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Tracking Errors')
    error_line_x, = ax3.plot([], [], 'r-', label='X Error')
    error_line_y, = ax3.plot([], [], 'g-', label='Y Error')
    error_line_z, = ax3.plot([], [], 'm-', label='Z Error')
    ax3.legend()
    ax3.grid(True)

    def update(frame):
        """
        This function updates each line for the given animation frame.
        We draw data from history up to 'frame' index.
        """
        t_data = history['time'][:frame]
        
        # Update altitude plot data
        altitude_line.set_data(t_data, history['y'][:frame])
        setpoint_line.set_data(t_data, history['setpoint_y'][:frame])
        
        # Update thrust plot data
        thrust_line_x.set_data(t_data, history['thrust_x'][:frame])
        thrust_line_y.set_data(t_data, history['thrust_y'][:frame])
        thrust_line_z.set_data(t_data, history['thrust_z'][:frame])
        
        # Update error plot data
        error_line_x.set_data(t_data, history['error_x'][:frame])
        error_line_y.set_data(t_data, history['error_y'][:frame])
        error_line_z.set_data(t_data, history['error_z'][:frame])
        
        return (altitude_line, setpoint_line, 
                thrust_line_x, thrust_line_y, thrust_line_z,
                error_line_x, error_line_y, error_line_z)

    # Create animation object. blit=True optimizes redrawing only changed artists.
    ani = FuncAnimation(fig, update, frames=len(history['time']),
                        interval=dt * 1000, blit=True)
    plt.tight_layout()
    return ani

# -------------------------------------
# Enhanced 3D Drone Flight Animation
# -------------------------------------
def animate_drone_3d(history, dt):
    """
    Create a 3D animation showing the drone position and orientation over time.
    We draw:
      - A marker for drone center
      - Two arms (X-axis and Z-axis) and four rotor positions (as red dots)
      - A trajectory line
      - A green 'X' at the current setpoint for reference
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine axis limits with margins so that the entire flight path is visible
    x_min, x_max = np.min(history['x']), np.max(history['x'])
    y_min, y_max = np.min(history['y']), np.max(history['y'])
    z_min, z_max = np.min(history['z']), np.max(history['z'])
    
    x_range = max(abs(x_min), abs(x_max))
    z_range = max(abs(z_min), abs(z_max))
    margin = max(x_range, z_range) * 0.2
    
    # We assume the ground plane is at y=0
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(0, y_max + 1)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (Alt, m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Drone Flight')
    
    # Draw a semi-transparent ground plane for context
    x_grid, z_grid = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, 10),
        np.linspace(z_min - margin, z_max + margin, 10)
    )
    y_grid = np.zeros_like(x_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='green')

    # Prepare empty plot objects; they will be updated each frame
    drone_center, = ax.plot([], [], [], 'ko', markersize=14)  # black circle for center
    drone_x_arm, = ax.plot([], [], [], 'k-', lw=2)           # black line for X-axis arm
    drone_z_arm, = ax.plot([], [], [], 'k-', lw=2)           # black line for Z-axis arm
    rotor1, = ax.plot([], [], [], 'ro', markersize=10)       # red dot for rotor 1
    rotor2, = ax.plot([], [], [], 'ro', markersize=10)
    rotor3, = ax.plot([], [], [], 'ro', markersize=10)
    rotor4, = ax.plot([], [], [], 'ro', markersize=10)
    
    # Trajectory line (blue, semi-transparent)
    trajectory, = ax.plot([], [], [], 'b-', alpha=0.7, lw=1.5)
    
    # Current setpoint marker (green X)
    setpoint, = ax.plot([], [], [], 'gx', markersize=10)

    def update(frame):
        """
        Update the 3D plot elements for frame index 'frame'.
        We draw the drone's body and its path up to the current time.
        """
        x = history['x'][frame - 1]
        y = history['y'][frame - 1]
        z = history['z'][frame - 1]
        
        # Update drone's center marker
        drone_center.set_data([x], [y])
        drone_center.set_3d_properties([z])
        
        d = 0.6  # half-length for visualizing the drone's arms (arbitrary scaling)
        
        # Update X-axis arm (horizontal line along X direction)
        drone_x_arm.set_data([x - d, x + d], [y, y])
        drone_x_arm.set_3d_properties([z, z])
        
        # Update Z-axis arm (horizontal line along Z direction)
        drone_z_arm.set_data([x, x], [y, y])
        drone_z_arm.set_3d_properties([z - d, z + d])
        
        # Update rotor markers at the ends of the arms
        # Rotor 1: front-left (-d on x-axis)
        rotor1.set_data([x - d], [y])
        rotor1.set_3d_properties([z])
        # Rotor 2: front-right (+d on x-axis)
        rotor2.set_data([x + d], [y])
        rotor2.set_3d_properties([z])
        # Rotor 3: back-left (-d on z-axis)
        rotor3.set_data([x], [y])
        rotor3.set_3d_properties([z - d])
        # Rotor 4: back-right (+d on z-axis)
        rotor4.set_data([x], [y])
        rotor4.set_3d_properties([z + d])
        
        # Update the trajectory line up to current index
        trajectory.set_data(history['x'][:frame], history['y'][:frame])
        trajectory.set_3d_properties(history['z'][:frame])
        
        # Show current setpoint as a green 'X'
        sp_x = history['setpoint_x'][frame - 1]
        sp_y = history['setpoint_y'][frame - 1]
        sp_z = history['setpoint_z'][frame - 1]
        setpoint.set_data([sp_x], [sp_y])
        setpoint.set_3d_properties([sp_z])
        
        return (drone_center, drone_x_arm, drone_z_arm, 
                rotor1, rotor2, rotor3, rotor4, 
                trajectory, setpoint)

    # Create the 3D animation. We set blit=False because 3D artists
    # often don't update correctly with blitting.
    ani = FuncAnimation(fig, update, frames=len(history['time']),
                        interval=dt * 1000, blit=False)
    return ani

# -------------------------------------
# Main:
# Run the simulation and launch the two separate animation windows
# -------------------------------------
if __name__ == "__main__":
    # Seed the random number generator for reproducibility of random waypoints
    np.random.seed(42)
    
    # Run the flight simulation for 3 cycles of 10 seconds each, with 0.05 s time steps
    history, dt = simulate_multi_cycle(num_cycles=3, T_cycle=10.0, dt=0.05)
    
    # Create and display the time-series animation (altitude, thrusts, errors)
    ani_time = animate_time_plots(history, dt)
    
    # Create and display the 3D flight animation
    ani_3d = animate_drone_3d(history, dt)
    
    plt.show()
