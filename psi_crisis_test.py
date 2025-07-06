# Re-import necessary libraries due to environment reset
import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 0.0573
epsilon = 0.02
U = 0.9
psi_xt = 0.7
phi_t = np.pi / 3
phi = 1.618  # golden ratio

# Functions
def eta_E(phi_val, u_val, lambda_3_val): # Renamed to avoid conflict with global 'phi'
    """
    Calculates the entropic stress-energy (η_E) on the Ψ-field.
    This metric quantifies dissipation and noise within the Ψ-field.
    """
    return C * (phi_val - u_val * lambda_3_val) ** 1.5 + epsilon

def delta_theta(x_val): # Renamed
    """
    Defines the phase coherence threshold (ΔΘ) based on recursive depth (x).
    A critical limit for phase harmony; exceeding it indicates decoherence.
    """
    return 3.6 - 7 * x_val ** (-0.5) - (U - (psi_xt * np.cos(phi_t)))

def run_crisis_test_plot():
    """
    Performs the crisis test scan and generates the coherence plot.
    """
    print("--- Running Ψ-Codex Crisis Test ---")
    # Crisis test: scan over ranges of x and λ₃
    x_vals_scan = np.linspace(0.5, 5, 100) # Renamed to avoid conflict with meshgrid X
    lambda_3_vals_scan = np.linspace(0.5, 1.8, 100) # Renamed

    # Prepare meshgrid
    X_grid, L3_grid = np.meshgrid(x_vals_scan, lambda_3_vals_scan) # Renamed meshgrid outputs

    # Calculate metrics on the grid
    # The global 'phi' is used here as intended in the original script for eta_E.
    # u (agentic resistance) is set to 0.8 for this simulation.
    eta_E_grid_calc = eta_E(phi, 0.8, L3_grid)
    delta_theta_grid_calc = delta_theta(X_grid)

    coherence_state_grid = eta_E_grid_calc < delta_theta_grid_calc # Boolean array

    print(f"Calculated eta_E grid min/max: {np.min(eta_E_grid_calc):.3f}/{np.max(eta_E_grid_calc):.3f}")
    print(f"Calculated delta_theta grid min/max: {np.min(delta_theta_grid_calc):.3f}/{np.max(delta_theta_grid_calc):.3f}")
    print(f"Coherent states proportion: {np.mean(coherence_state_grid):.2%}")

    # Plotting the crisis map
    # This part is conceptual for an environment without direct display.
    # It would generate a plot if matplotlib can render.
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Using contourf to plot zones based on coherence_state_grid (True/False)
        # levels are set to define boundaries for True (1) and False (0)
        c = ax.contourf(X_grid, L3_grid, coherence_state_grid.astype(int),
                        levels=[-0.5, 0.5, 1.5], colors=['#f8d2d2', '#d2f8d2']) # red for False, green for True

        ax.set_title('Ψ-Codex Crisis Test: Coherence vs Decoherence Zones')
        ax.set_xlabel('Recursive Depth x')
        ax.set_ylabel('λ₃ (Resilience Factor)')

        # Create legend handles
        # Using Patch objects for solid color legend items might be clearer for contourf.
        import matplotlib.patches as mpatches
        coherent_patch = mpatches.Patch(color='#d2f8d2', label='🟢 Coherent')
        decoherence_patch = mpatches.Patch(color='#f8d2d2', label='🔴 Decoherent')
        ax.legend(handles=[coherent_patch, decoherence_patch])

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout to prevent overlapping elements

        # Attempt to save the figure instead of showing it directly.
        plot_filename = "psi_crisis_test_coherence_map.png"
        plt.savefig(plot_filename)
        print(f"Crisis test plot saved to {plot_filename}")
        # plt.show() # plt.show() would block in many non-interactive environments.

    except Exception as e:
        print(f"Matplotlib plot generation/saving failed: {e}")
        print("Ensure matplotlib is installed and a backend is available if running locally.")
        print("In this environment, the plot image won't be displayed directly but calculations are done.")

    print("--- Ψ-Codex Crisis Test Complete ---")

if __name__ == '__main__':
    # This script runs its main logic at the global level if plt.show() is active.
    # Encapsulating in a function and calling it makes it cleaner.
    run_crisis_test_plot()

```
