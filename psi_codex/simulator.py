import math
import numpy as np
import matplotlib.pyplot as plt
import yaml
from .pdf_generator import generate_pdf

# --- Custom Exception for Gödelian Collapse ---
class GodelianCollapse(Exception):
    """Custom exception for Gödelian paradox or coherence collapse."""
    pass

# --- Core Functions from the original notebook ---

def eta_E(phi: float, u: float, lambda_3: float, current_epsilon: float, C: float = 0.0573) -> float:
    """Computes the Stress-Energy Metric (η_E)."""
    P_val = phi - u * lambda_3
    return C * (abs(P_val)**1.5) + current_epsilon

def delta_theta(x: float, current_U: float) -> float:
    """Calculates the Phase Coherence Threshold (ΔΘ(x))."""
    if x <= 0:
        return float('inf')
    return 3.6 - 7 * (x**-0.5) - current_U

def factorial_mod_255(X: float) -> int:
    """Computes ψ' = factorial(int(abs(X))) % 255."""
    x_int_abs = int(abs(X))
    if x_int_abs >= 17:
        return 0
    if x_int_abs < 2:
        return 1
    return math.factorial(x_int_abs) % 255

def detect_godelian_failure(phi_val: float, x_val: float, delta_t_val: float, temperature_T: float):
    """Detects Gödelian Collapse or Quantum Coherence Collapse."""
    if x_val > 2.8 and phi_val > 1.6:
        raise GodelianCollapse("Gödelian Paradox detected")

    coherence_length = 1 / (temperature_T**0.5) if temperature_T > 0 else float('inf')
    if delta_t_val > coherence_length:
        raise GodelianCollapse(f"Quantum Coherence Collapse (ΔΘ={delta_t_val:.3f} > L_coh={coherence_length:.3f})")

# --- Main Simulation Logic ---

def run_simulation():
    """
    Runs the main Ψ-Codex simulation, generates plots, and creates a PDF report.
    """
    print("Starting Ψ-Codex simulation...")

    # Load simulation parameters from config file
    with open("psi-codex-config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sim_params = config.get("simulation", {})
    x_min = sim_params.get("x_min", 0.1)
    x_max = sim_params.get("x_max", 5.0)
    x_steps = sim_params.get("x_steps", 100)
    lambda_3 = sim_params.get("lambda_3", 0.9)
    epsilon = sim_params.get("epsilon", 0.05)
    temperature_T = sim_params.get("temperature_T", 0.5)
    phi_val = sim_params.get("phi_val", 1.618)
    U_val = sim_params.get("U_val", 0.1)

    x_values = np.linspace(x_min, x_max, x_steps)

    eta_e_values = []
    delta_t_values = []
    status_points = []

    for x in x_values:
        eta = eta_E(phi_val, U_val, lambda_3, epsilon)
        delta_t = delta_theta(x, U_val)
        eta_e_values.append(eta)
        delta_t_values.append(delta_t)

        try:
            detect_godelian_failure(phi_val, x, delta_t, temperature_T)
            if eta >= delta_t:
                status_points.append((x, max(eta, delta_t), 'decoherence'))
        except GodelianCollapse:
            status_points.append((x, max(eta, delta_t), 'collapse'))

    eta_e_values = np.array(eta_e_values)
    delta_t_values = np.array(delta_t_values)

    # --- Plotting ---
    print("Generating simulation plots...")
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, eta_e_values, 'r--', label='η_E (Stress-Energy)')
    plt.plot(x_values, delta_t_values, 'k-', label='ΔΘ(x) (Coherence Threshold)')

    coherence_length = 1 / (temperature_T**0.5) if temperature_T > 0 else float('inf')
    plt.axhline(y=coherence_length, color='b', linestyle=':', label=f'L_coh (Coherence Length) = {coherence_length:.2f}')

    # Plot decoherence and collapse points
    decoherence_added = False
    collapse_added = False
    for x, y, status in status_points:
        if status == 'decoherence':
            plt.plot(x, y, 'yo', markersize=8, label='Decoherence Point' if not decoherence_added else "")
            decoherence_added = True
        elif status == 'collapse':
            plt.plot(x, y, 'ro', markersize=8, label='Collapse Point' if not collapse_added else "")
            collapse_added = True


    plt.title("Ψ-Codex: Coherence Dynamics")
    plt.xlabel("Recursive Depth (x)")
    plt.ylabel("Threshold Values")

    # Fill stable and unstable regions
    plt.fill_between(x_values, delta_t_values, eta_e_values,
                     where=(eta_e_values >= delta_t_values),
                     color='yellow', alpha=0.3, label='Decoherence Region')

    plt.fill_between(x_values, delta_t_values, coherence_length,
                        where=(delta_t_values >= coherence_length),
                        color='red', alpha=0.3, label='Collapse Region')

    plt.fill_between(x_values, 0, np.minimum(delta_t_values, coherence_length),
                     color='green', alpha=0.2, label='Stable Region')


    # Remove duplicate labels from legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    plt.ylim(bottom=0)


    plot_filename = "psi_critical_dynamics_enhanced_fixed_points.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()

    # --- PDF Generation ---
    print("Generating PDF report...")
    generate_pdf()

if __name__ == "__main__":
    run_simulation()
