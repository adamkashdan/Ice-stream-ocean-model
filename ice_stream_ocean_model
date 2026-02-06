"""
2 Directional Coupled Ice Stream Ocean Model
Logan E. Mann // Ice and Climate Group Georgia Tech
Converted from MATLAB to Python
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants
YEAR_CON = 3600 * 24 * 365  # Seconds in a year (s)


class Parameters:
    """Container for all model parameters"""
    def __init__(self):
        # Primary Parameters
        self.Ts = 273.15 - 38  # K
        self.G = 0.04  # W m-2
        self.xi = 10  # Heinrich Salinity Flux Parameter
        self.mf = 10 / YEAR_CON  # Submarine Melt Rate (m/s)
        self.gamma = 260 / (3e5 * YEAR_CON)  # Relaxation Time Parameter
        
        # Initial Conditions
        self.h_init = 900  # Initial avg ice stream height (m)
        self.e_init = 0.6  # Initial Void Ratio
        self.Zs_init = 1  # Initial Till Height (m)
        self.Tb_init = 273.15  # Initial Basal Temperature (K)
        self.L_init = 1000e3  # Initial Grounding Line Position (m)
        self.x_init = 0.2  # Initial Ocean Temperature (nondimensional)
        self.y_init = 0.2  # Salinity (nondimensional)
        
        # Time parameters
        self.t_final = 3e5  # Total time of integration (yr)
        self.tspan = [0, YEAR_CON * self.t_final]
        
        # Ocean Oscillator Parameters
        self.T_0 = 5  # Temperature of NADW (K)
        self.T_A = -15  # Relaxation Temperature
        self.delta = 0.001
        self.epsilon = -0.01
        self.nu1 = 0.1
        self.nu2 = 5
        
        # Ice Stream Parameters
        self.a = 9.44e8  # Till empirical coefficient (Pa)
        self.ac = 0.1 / YEAR_CON  # Accumulation rate (m sec-1)
        self.Ag = 5e-25  # Glen's law rate factor (Pa-3 s-1)
        self.b = 21.7  # Till empirical exponent (unitless)
        self.Ci = 1.94e6  # Volumetric heat capacity of ice (J K-1 m-3)
        self.ec = 0.3  # Till consolidation threshold (unitless)
        self.g = 9.81  # Acceleration (s-2)
        self.hb = 3  # Thickness of temperate ice layer (m)
        self.ki = 2.1  # Thermal conductivity of ice (J s-1 m-1 K-1)
        self.Lf = 3.35e5  # Specific latent heat of ice (J kg-1)
        self.n = 3  # Glen's law exponent (unitless)
        self.W = 40e3  # Ice stream trunk width (m)
        self.ws = 1  # Till saturation threshold (m)
        self.Zs_min = 1e-3  # minimum till thickness
        self.rhoi = 917  # Ice Density (kg m-3)
        self.rhow = 1028  # Seawater density (kg m-3)
        self.Tm = 273.15  # Melting point of water (K)
        
        # Grounding Line Parameters
        self.f = 0.4  # Coulomb Friction coefficient
        
        # Bed Parameters
        self.b0 = 100  # Ice divide bed height(m)
        self.bx = -5e-4  # Prograde bed slope


def rhs_function(t, X, p):
    """
    Right-hand side function for the coupled ice stream-ocean model
    
    Parameters:
    -----------
    t : float
        Time
    X : array
        State vector [h, e, Zs, Tb, L, x, y]
    p : Parameters
        Parameter object
    
    Returns:
    --------
    rhs : array
        Derivatives [dhdt, dedt, dZsdt, dTbdt, dLdt, dxdt, dydt]
    """
    
    # Unpack state variables
    h = X[0]
    e = X[1]
    Zs = max(min(X[2], p.Zs_init), p.Zs_min)
    Tb = min(X[3], p.Tm)
    L = X[4]
    x = X[5]
    y = X[6]
    
    # Diagnostic Equations
    # Bed Equations
    bg = p.b0 + p.bx * L
    hg = -p.rhow / p.rhoi * bg
    
    # Basic Model Equations
    e = max(e, p.ec)
    taub = p.a * np.exp(-p.b * (e - p.ec))
    taud = p.rhoi * p.g * h**2 / L
    ub = (p.Ag / 256) * p.W**(p.n + 1) / (4**p.n * (p.n + 1) + h**p.n) * max(taud - taub, 0)**p.n
    
    # Grounding Line Equations
    Qg = 0.61 * (8 * p.Ag * (p.rhoi * p.g)**p.n) / (4**p.n * p.f) * \
         (1 - p.rhoi / p.rhow)**(p.n - 1) * hg**(p.n + 2)
    Qd = (2 * p.Ag / (p.n + 2) * h**2 * min(taub, taud)**p.n)
    Qv = ub * h
    Q = Qv + Qd
    
    # Ocean Forcing
    mu1 = p.nu1 / (1 + p.nu1) + p.epsilon * p.nu1
    mu2 = p.nu2 / (1 + p.nu2) + p.epsilon * p.nu2
    mu = mu2 - p.delta * p.nu2
    
    if y - x <= p.epsilon:
        nu = p.nu1
    else:
        nu = p.nu2
    
    To = max(x * (p.T_A - p.T_0) + p.T_0, 0)
    Qm = p.mf * To * (-bg)
    Qg = Qg + Qm  # Add Melt Flux to GLine flux
    
    # Prognostic Equations
    if ((Zs == p.Zs_min and Tb == 273.15 and 
         ((taub * ub) + p.G + (p.ki * (p.Ts - Tb) / h)) < 0) or 
        (Zs == p.Zs_min and Tb < 273.15)):
        # Till is frozen
        ub = 0
        dedt = 0
        dZsdt = 0
        dTbdt = (1 / (p.hb * p.Ci)) * ((taub * ub) + p.G + (p.ki * (p.Ts - Tb) / h))
    elif ((e == p.ec and Zs == p.Zs_init and 
           ((taub * ub) + p.G + (p.ki * (p.Ts - Tb) / h)) < 0) or 
          (e == p.ec and Zs < p.Zs_init)):
        ub = 0
        dedt = 0
        dZsdt = ((taub * ub) + p.G + (p.ki * (p.Ts - Tb) / h)) / (p.Lf * p.rhoi)
        dTbdt = 0
    else:
        dedt = ((taub * ub) + p.G + (p.ki * (p.Ts - Tb) / h)) / (Zs * p.Lf * p.rhoi)
        dZsdt = 0
        dTbdt = 0
    
    dhdt = p.ac - Qg / L - h * (Q - Qg) / (hg * L)
    dLdt = (Q - Qg) / hg
    dxdt = (1 - x - nu * x) * p.gamma
    dydt = (mu * (1 - p.xi * Qg) - nu * y) * p.gamma
    
    # Progress indicator
    if int(100 * t / p.tspan[1]) % 10 == 0:
        print(f'Percent Integration Complete: {100 * t / p.tspan[1]:.1f}%')
    
    return [dhdt, dedt, dZsdt, dTbdt, dLdt, dxdt, dydt]


def main():
    """Main execution function"""
    
    # Initialize parameters
    p = Parameters()
    
    # Initial conditions
    init = [p.h_init, p.e_init, p.Zs_init, p.Tb_init, p.L_init, p.x_init, p.y_init]
    
    # Solve ODE
    print("Starting integration...")
    sol = solve_ivp(
        lambda t, X: rhs_function(t, X, p),
        p.tspan,
        init,
        method='DOP853',  # Similar to MATLAB's ode113
        rtol=1e-9,
        atol=1e-9,
        dense_output=True
    )
    
    # Extract solution
    time = sol.t
    soln = sol.y.T
    
    h = soln[:, 0]
    e = soln[:, 1]
    Zs = soln[:, 2]
    Tb = soln[:, 3]
    L = soln[:, 4]
    x = soln[:, 5]
    y = soln[:, 6]
    
    # Diagnostic Equations
    bg = p.b0 + p.bx * L
    hg = -p.rhow / p.rhoi * bg
    
    e = np.maximum(e, p.ec)
    taub = np.minimum(p.a * np.exp(-p.b * e), p.f * (p.rhoi * p.g * h))
    taud = p.rhoi * p.g * h**2 / L
    ub = (p.Ag / 256) * p.W**(p.n + 1) / (4**p.n * (p.n + 1) + h**p.n) * \
         np.maximum(taud - taub, 0)**p.n
    
    Qg = 0.61 * (8 * p.Ag * (p.rhoi * p.g)**p.n) / (4**p.n * p.f) * \
         (1 - p.rhoi / p.rhow)**(p.n - 1) * hg**(p.n + 2)
    Qd = (2 * p.Ag / (p.n + 2) * h**2 * np.minimum(taub, taud)**p.n)
    Qv = ub * h
    Q = Qv + Qd
    
    # Ocean forcing
    nu = np.where(y - x <= p.epsilon, p.nu1, p.nu2)
    To = np.maximum(x * (p.T_A - p.T_0) + p.T_0, 0)
    Qm = p.mf * To * (-bg)
    Qg = Qg + Qm
    
    # Plotting
    plot_results(time, h, L, e, Q, Qg, ub, hg, Tb, bg, To, Qm, p)
    
    # Phase analysis
    phase_analysis(time, h, To, p)
    
    plt.show()
    
    return time, soln, p


def plot_results(time, h, L, e, Q, Qg, ub, hg, Tb, bg, To, Qm, p):
    """Plot main results"""
    
    # Figure 1: Main state variables
    fig1, axes = plt.subplots(5, 2, figsize=(12, 15))
    
    axes[0, 0].plot(time / YEAR_CON, h / 1000)
    axes[0, 0].set_xlabel('time (yr)')
    axes[0, 0].set_ylabel('h (km)')
    
    axes[0, 1].plot(time / YEAR_CON, L / 1000)
    axes[0, 1].set_xlabel('time (yr)')
    axes[0, 1].set_ylabel('L (km)')
    
    axes[1, 0].plot(time / YEAR_CON, e)
    axes[1, 0].set_xlabel('time (yr)')
    axes[1, 0].set_ylabel('e')
    
    axes[1, 1].plot(time / YEAR_CON, Q)
    axes[1, 1].set_xlabel('time (yr)')
    axes[1, 1].set_ylabel('Q')
    
    axes[2, 0].plot(time / YEAR_CON, Qg)
    axes[2, 0].set_xlabel('time (yr)')
    axes[2, 0].set_ylabel('Qg')
    
    axes[2, 1].plot(time / YEAR_CON, ub * YEAR_CON / 1000)
    axes[2, 1].set_xlabel('time (yr)')
    axes[2, 1].set_ylabel('ub (km/yr)')
    
    axes[3, 0].plot(time / YEAR_CON, hg)
    axes[3, 0].set_xlabel('time (yr)')
    axes[3, 0].set_ylabel('h_g (m)')
    
    axes[3, 1].plot(time / YEAR_CON, Tb)
    axes[3, 1].set_xlabel('time (yr)')
    axes[3, 1].set_ylabel('Tb')
    
    axes[4, 0].plot(time / YEAR_CON, bg)
    axes[4, 0].set_xlabel('time (yr)')
    axes[4, 0].set_ylabel('b_g')
    
    axes[4, 1].axis('off')
    
    plt.tight_layout()
    
    # Figure 2: Ocean forcing
    fig2, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    axes[0].plot(time / YEAR_CON, To)
    axes[0].set_xlabel('time (yr)')
    axes[0].set_ylabel('T_o (°C)')
    
    axes[1].plot(time / YEAR_CON, L / 1000)
    axes[1].set_xlabel('time (yr)')
    axes[1].set_ylabel('L (km)')
    
    axes[2].plot(time / YEAR_CON, Qm * YEAR_CON)
    axes[2].set_xlabel('time (yr)')
    axes[2].set_ylabel('Q_m (m²/a)')
    
    fig2.suptitle(f'G={p.G} W/m², T_s={p.Ts - 273.15}°C, ξ={p.xi}, m_f={p.mf * YEAR_CON} m/yr°C')
    plt.tight_layout()


def phase_analysis(time, h, To, p):
    """Perform phase analysis between Heinrich events and DO events"""
    
    # Calculate dhdt
    dhdt = np.diff(h) / np.diff(time)
    dhdt_thresh = np.where(dhdt > np.min(dhdt) - np.min(dhdt) * 3 / 4, 0, dhdt)
    
    # Find peaks in -dhdt (Heinrich events)
    peaks_h, _ = find_peaks(-dhdt)
    phi_H = peaks_h[1:]  # Remove first peak
    
    # Plot dhdt with peaks
    fig5 = plt.figure(figsize=(10, 5))
    plt.plot(time[:-1] / YEAR_CON, dhdt)
    plt.plot(time[phi_H] / YEAR_CON, dhdt[phi_H], 'x')
    plt.xlabel('time (yr)')
    plt.ylabel('dh/dt')
    plt.title('Rate of change of ice height')
    
    # Find peaks in To (DO events)
    phi_DO_all, _ = find_peaks(To)
    
    # Match DO peaks to Heinrich events
    phi_DO = np.zeros(len(phi_H), dtype=int)
    for indx in range(len(phi_H)):
        phi_DO_temp = phi_DO_all[phi_DO_all < phi_H[indx]]
        if len(phi_DO_temp) == 0:
            phi_DO_temp = np.array([0])
        phi_DO[indx] = phi_DO_temp[-1]
    
    dphi = time[phi_H] - time[phi_DO]  # Phase difference
    
    # Calculate periods
    if len(phi_DO_all) > 1:
        T_DO = np.mean(np.diff(time[phi_DO_all]))
    else:
        T_DO = 0
    
    if len(phi_H) > 1:
        T_H = np.mean(np.diff(time[phi_H]))
    else:
        T_H = 0
    
    # Figure 3: Phase analysis
    extent = 100
    fig3, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top panel
    ax1 = axes[0]
    ax1_right = ax1.twinx()
    
    ax1.plot(time / YEAR_CON / 1000, h, 'b')
    ax1.plot(time[phi_H] / YEAR_CON / 1000, h[phi_H], 'xk')
    ax1.set_xlabel('time (kyr)')
    ax1.set_ylabel('h (m)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1_right.plot(time / YEAR_CON / 1000, To, 'r', linewidth=1)
    ax1_right.plot(time[phi_DO] / YEAR_CON / 1000, To[phi_DO], 'xr')
    ax1_right.set_ylabel('T_o', color='r')
    ax1_right.tick_params(axis='y', labelcolor='r')
    ax1_right.set_ylim([0, 2])
    ax1.set_xlim([0, extent])
    
    # Bottom panel
    axes[1].plot(time[phi_H] / YEAR_CON / 1000, dphi / YEAR_CON, '.', markersize=10)
    axes[1].set_xlabel('time (kyr)')
    axes[1].set_ylabel('Φ_h - Φ_{T_o} (Phase Difference in years)')
    axes[1].set_xlim([0, extent])
    if len(phi_H) > 1:
        axes[1].set_ylim([np.min(dphi / YEAR_CON), np.max(dphi / YEAR_CON)])
    
    if len(phi_H) > 1 and T_DO > 0:
        ratio = (time[phi_H[-1]] - time[phi_H[-2]]) / T_DO
        fig3.suptitle(f'{ratio:.1f}:1, ξ={p.xi}, m_f={p.mf * YEAR_CON} m/yr°C')
    
    plt.tight_layout()


if __name__ == "__main__":
    time, solution, params = main()
