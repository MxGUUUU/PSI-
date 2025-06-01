import numpy as np
import math # For math.factorial
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# from scipy.special import airy # Not used by user's initialize_field in this version
from scipy.special import factorial # For PsiCodexSimulator._factorial_mod_255
from fpdf import FPDF
from datetime import datetime
import os
import json

# --- Fundamental Constants & Parameters ---
h_σ = 1/16  # Planck-like constant for Ψ-field (Ising model conformal weight)
C = 0.0573  # Speed of light / information propagation in Ψ-field
Λ_CDM = 0.3 # Cosmological Constant (Dark Energy density fraction) - conceptual

class DarkMatterConstraints:
    def __init__(self, density_param=0.25, interaction_xsection=0.01, lambda3=0.5):
        self.density_parameter = density_param
        self.interaction_cross_section = interaction_xsection
        self.lambda3 = lambda3  # λ₃ coupling for higher-order Ψ interactions
        print(f"DarkMatterConstraints: Ω_DM={self.density_parameter}, σ_DM-Ψ={self.interaction_cross_section}, λ₃={self.lambda3}")

    def get_potential_term(self, psi_field_amplitude_sq):
        return -self.density_parameter * self.interaction_cross_section * psi_field_amplitude_sq

    def check_stability(self, Ψ_field_spatial, lambda3_effective, x_spatial_domain_mean): # x_range.mean() is passed here
        eta_E_conceptual = np.mean(np.abs(Ψ_field_spatial)**2) * (1 + lambda3_effective) / (1 + x_spatial_domain_mean**2)
        phases = np.angle(Ψ_field_spatial)
        unwrapped_phases = np.unwrap(phases)
        if len(unwrapped_phases) < 2:
            delta_Theta_conceptual = 0
        else:
            delta_Theta_conceptual = np.std(unwrapped_phases) * (1 / (h_σ * C))
        delta_Theta_conceptual = np.abs(delta_Theta_conceptual)
        critical_eta_E = 1/8
        coherence_limit_dTheta = 3.6
        stable_eta_E = eta_E_conceptual < critical_eta_E
        stable_delta_Theta = delta_Theta_conceptual < coherence_limit_dTheta
        is_overall_stable = stable_eta_E and stable_delta_Theta
        if len(unwrapped_phases) < 2:
             holonomy_conceptual = 0
        else:
            holonomy_conceptual = np.sum(np.diff(unwrapped_phases)) / (2 * np.pi)
        return is_overall_stable, holonomy_conceptual, eta_E_conceptual, delta_Theta_conceptual

class E8_Z4_Lattice:
    def __init__(self, scale=1.0, n_attractors=8):
        np.random.seed(42)
        self.attractors = (np.random.rand(n_attractors, 4) - 0.5) * 2 * scale
        self.attractors[0,:] = [1,0,0,0]
        print(f"E8_Z4_Lattice (ODE context): Initialized with {n_attractors} attractors, scale={scale}")

    def apply_constraint_to_ode_state(self, y_ode_state_current):
        distances = np.linalg.norm(self.attractors - y_ode_state_current, axis=1)
        nearest_idx = np.argmin(distances)
        target_attractor = self.attractors[nearest_idx]
        y_constrained = y_ode_state_current + 0.05 * (target_attractor - y_ode_state_current)
        return y_constrained, nearest_idx

# --- Conceptual a₂ Coefficient Calculation ---
def calculate_a2_coefficient(x_spatial_domain, psi_field_spatial, y_i_proxy_values):
    if not (isinstance(x_spatial_domain, np.ndarray) and
            isinstance(psi_field_spatial, np.ndarray) and
            isinstance(y_i_proxy_values, np.ndarray)):
        raise TypeError("Inputs x_spatial_domain, psi_field_spatial, and y_i_proxy_values must be numpy arrays.")
    if not (x_spatial_domain.shape == psi_field_spatial.shape == y_i_proxy_values.shape):
        raise ValueError("Input arrays must have the same shape for element-wise operations.")
    psi_star = np.conjugate(psi_field_spatial)
    denominator = (x_spatial_domain**5) * psi_star * y_i_proxy_values - 1.0
    a2_field = np.full_like(denominator, np.nan, dtype=complex)
    epsilon_div_zero = 1e-12
    valid_mask = np.abs(denominator) > epsilon_div_zero
    if np.any(valid_mask):
        a2_field[valid_mask] = 3.0 / denominator[valid_mask]
    return a2_field

class PsiCodexSimulator:
    def __init__(self, dm_constraints, e8_lattice, time_params, x_range_params, lambda3_resilience_factor=1.0):
        self.dm_constraints = dm_constraints
        self.e8_lattice = e8_lattice
        self.t_span = (time_params['start'], time_params['end'])
        self.t_eval_count = time_params['points']
        self.x_range = np.linspace(x_range_params['min'], x_range_params['max'], x_range_params['points'])
        self.Ψ = np.zeros_like(self.x_range, dtype=complex)
        self.λ3 = dm_constraints.lambda3 * lambda3_resilience_factor
        self.current_psi_state_ode = np.array([0.1, 0.1, 0.0, 0.0])
        self.stability_history = []
        self.collapse_events = []
        print("PsiCodexSimulator (User Script Version) Initialized.")

    def initialize_field(self):
        phi_initial = np.pi / 4
        self.Ψ = np.sin(np.pi * self.x_range / (self.x_range[-1] - self.x_range[0]) + phi_initial) * \
                   np.exp(1j * np.pi/3 * np.tanh(self.x_range))
        self.Ψ = self.Ψ / np.max(np.abs(self.Ψ)) if np.max(np.abs(self.Ψ)) > 0 else self.Ψ
        print(f"  Initial spatial field Ψ(x) created. Mean |Ψ|={np.mean(np.abs(self.Ψ)):.3f}")
        self.current_psi_state_ode[0] = np.mean(np.real(self.Ψ))
        self.current_psi_state_ode[1] = np.mean(np.imag(self.Ψ))
        self.current_psi_state_ode[2:] = [0.0,0.0]

    def _psi_field_ode_method(self, t, y_ode_state):
        psi_r_avg, psi_i_avg, dpsi_r_dt_avg, dpsi_i_dt_avg = y_ode_state
        psi_amplitude_sq_avg = psi_r_avg**2 + psi_i_avg**2
        k_eff_sq = (np.pi / (self.x_range[-1] - self.x_range[0]))**2
        dm_term_coeff = -self.dm_constraints.density_parameter * \
                        self.dm_constraints.interaction_cross_section * \
                        (1 + self.λ3 * psi_amplitude_sq_avg)
        ddpsi_r_dt2 = -k_eff_sq * C**2 * psi_r_avg + dm_term_coeff * psi_r_avg
        ddpsi_i_dt2 = -k_eff_sq * C**2 * psi_i_avg + dm_term_coeff * psi_i_avg
        return [dpsi_r_dt_avg, dpsi_i_dt_avg, ddpsi_r_dt2, ddpsi_i_dt2]

    def recursive_update(self):
        self.Ψ = self.Ψ + 0.01 * self.Ψ * (1 - np.abs(self.Ψ)**2)
        grad_real = np.gradient(np.real(self.Ψ), self.x_range, edge_order=2)
        grad_imag = np.gradient(np.imag(self.Ψ), self.x_range, edge_order=2)
        laplacian_real = np.gradient(grad_real, self.x_range, edge_order=2)
        laplacian_imag = np.gradient(grad_imag, self.x_range, edge_order=2)
        self.Ψ = self.Ψ + 0.005 * (laplacian_real + 1j*laplacian_imag)
        self.current_psi_state_ode, _ = self.e8_lattice.apply_constraint_to_ode_state(self.current_psi_state_ode)
        avg_ode_amplitude = np.sqrt(self.current_psi_state_ode[0]**2 + self.current_psi_state_ode[1]**2)
        current_spatial_norm = np.mean(np.abs(self.Ψ))
        if current_spatial_norm < 1e-9: current_spatial_norm = 1.0
        self.Ψ = self.Ψ * (avg_ode_amplitude / current_spatial_norm)

    def factorial_mod_255(self, n_val):
        n_int = int(n_val)
        if n_int < 0: return 0
        if n_int == 0: return 1
        if n_int >= 17: return 0
        return math.factorial(n_int) % 255

    def apply_shadow_integration(self):
        field_energy_metric = np.mean(np.abs(self.Ψ)**2) * 100
        x_transformed = self.factorial_mod_255(int(np.clip(field_energy_metric, 0, 50)))
        phase_factor = np.exp(1j * np.pi * x_transformed / 255.0)
        self.Ψ = self.Ψ * phase_factor
        event_info = {'type': 'shadow_integration', 'x_transformed': x_transformed, 'energy_metric': field_energy_metric}
        self.collapse_events.append(event_info)
        print(f"    Shadow Integration applied: X_transformed={x_transformed}, EnergyMetric={field_energy_metric:.2f}")

    def run_simulation(self, steps=10):
        self.initialize_field()
        self.stability_history = []
        self.collapse_events = []

        # The user's "yes integrate..." script's run_simulation structure:
        # It does not use solve_ivp within the main 'steps' loop.
        # 'steps' controls iterations of recursive_update and check_stability.
        # The ODE part seems to be handled differently or was conceptual for the avg state.
        # The current subtask is to add a2 logging to *this* specific loop structure.

        for step in range(steps):
            self.recursive_update() # This method updates self.Ψ, the spatial field

            stability, holonomy, η_E, ΔΘ = self.dm_constraints.check_stability(
                self.Ψ, self.λ3, self.x_range.mean() # Using self.λ3 as effective lambda3
            )

            x_domain_for_a2 = self.x_range
            current_psi_field_for_a2 = self.Ψ
            y_i_proxy_values_for_a2 = np.sin(x_domain_for_a2**2 + 3)
            a2_field = calculate_a2_coefficient(x_domain_for_a2, current_psi_field_for_a2, y_i_proxy_values_for_a2)

            a2_mean_abs = np.nanmean(np.abs(a2_field))
            a2_mean_real = np.nanmean(np.real(a2_field))
            a2_std_abs = np.nanstd(np.abs(a2_field))
            idx_resonance = np.argmin(np.abs(x_domain_for_a2 - 2.56))
            val_at_resonance = a2_field[idx_resonance]
            a2_at_resonance_real = np.real(val_at_resonance) if not np.isnan(val_at_resonance) else np.nan
            a2_at_resonance_imag = np.imag(val_at_resonance) if not np.isnan(val_at_resonance) else np.nan

            current_log_entry = {
                'step': step,
                'stability': bool(stability),
                'holonomy': holonomy,
                'η_E': η_E,
                'ΔΘ': ΔΘ,
                'psi_mean_abs': np.mean(np.abs(self.Ψ)),
                'a2_mean_abs': a2_mean_abs,
                'a2_mean_real': a2_mean_real,
                'a2_std_abs': a2_std_abs,
                'a2_at_resonance_real': a2_at_resonance_real,
                'a2_at_resonance_imag': a2_at_resonance_imag
            }
            self.stability_history.append(current_log_entry)

            if not stability:
                print(f"  Instability at step {step}: η_E={η_E:.3f}, ΔΘ={ΔΘ:.3f}. Applying shadow integration.")
                self.apply_shadow_integration()

        return self.stability_history, self.collapse_events

# --- Plotting Function (from user script) ---
def plot_results(history, x_range_domain, final_psi_spatial, filename="psi_codex_simulation_plots.png"):
    if not history: print("No history data to plot."); return
    steps = [log['step'] for log in history]
    psi_mean_abs = [log['psi_mean_abs'] for log in history]
    stability = [log['stability'] for log in history]
    holonomy_hist = [log.get('holonomy', np.nan) for log in history]
    eta_E_hist = [log.get('η_E', np.nan) for log in history]
    delta_Theta_hist = [log.get('ΔΘ', np.nan) for log in history]
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle("Psi-Codex Simulation Summary (User Script Version)", fontsize=16)
    axs[0].plot(steps, psi_mean_abs, label='Mean(|Ψ_spatial|)', color='green', marker='.')
    axs[0].set_ylabel('Mean |Ψ(x)|'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(steps, stability, label='Stability (1=Stable)', color='purple', marker='o', linestyle='-')
    axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(['Unstable', 'Stable']); axs[1].set_ylabel('Stability Status'); axs[1].legend(); axs[1].grid(True)
    ax2_twin = axs[2].twinx()
    axs[2].plot(steps, eta_E_hist, 'o-', color='blue', label='η_E (Stress-Energy)')
    ax2_twin.plot(steps, delta_Theta_hist, 's-', color='red', label='ΔΘ (Phase Coherence)')
    axs[2].set_ylabel('η_E Value', color='blue'); ax2_twin.set_ylabel('ΔΘ Value', color='red')
    axs[2].tick_params(axis='y', labelcolor='blue'); ax2_twin.tick_params(axis='y', labelcolor='red')
    axs[2].set_title('Conceptual Stability Metrics (η_E & ΔΘ)'); axs[2].legend(loc='upper left'); ax2_twin.legend(loc='upper right'); axs[2].grid(True)
    if final_psi_spatial is not None and x_range_domain is not None:
        axs[3].plot(x_range_domain, np.real(final_psi_spatial), label='Re(Ψ_final(x))', color='navy'); axs[3].plot(x_range_domain, np.imag(final_psi_spatial), label='Im(Ψ_final(x))', color='darkred', linestyle='--'); axs[3].set_xlabel('Spatial Domain (x)'); axs[3].set_ylabel('Ψ(x) Amplitude'); axs[3].set_title(f'Final Spatial Ψ-Field Distribution (After {len(history)} steps)'); axs[3].legend(); axs[3].grid(True)
    else:
        axs[3].text(0.5, 0.5, "No final spatial field data", ha='center', va='center'); axs[3].set_xlabel('Spatial Domain (x)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close(); print(f"Plots saved to {filename}")

# --- PDF Reporting Class (from user script) ---
class SimulationReport(FPDF):
    def header(self): self.set_font('Arial','B',12); self.cell(0,10,'Psi-Codex Simulation Report (User Version)',0,1,'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial','I',8); self.cell(0,10,f'Page {self.page_no()}/{{nb}}',0,0,'C')
    def chapter_title(self,title_str): self.set_font('Arial','B',12); self.set_fill_color(220,220,255); self.cell(0,6,title_str,0,1,'L',1); self.ln(4)
    def chapter_body(self,body_str): self.set_font('Times','',11); self.multi_cell(0,5,body_str); self.ln()
    def add_plot_image(self,img_path,caption=""):
        self.add_page(); self.chapter_title(caption if caption else "Plot")
        page_w = self.w - 2*self.l_margin
        try: self.image(img_path,x=None,y=None,w=page_w*0.9)
        except Exception as e: self.set_text_color(255,0,0); self.multi_cell(0,10,f"Error: {e}"); self.set_text_color(0,0,0)
        self.ln(2)

# --- Main Execution Block (from user script) ---
if __name__ == "__main__":
    print("Initiating Psi-Codex Simulation (User Script Version)...")
    dm_constraints_obj = DarkMatterConstraints(density_param=0.23, interaction_xsection=0.018, lambda3=0.52)
    e8_lattice_obj = E8_Z4_Lattice(scale=1.6, n_attractors=7) # Matches user script

    time_params_dict = {'start':0, 'end':20, 'points':100} # Not directly used by run_simulation's outer loop in this version
    x_params_dict = {'min': -8, 'max': 8, 'points':256}
    num_iter_steps = 50 # This is the 'steps' for run_simulation outer loop

    simulator_obj = PsiCodexSimulator(dm_constraints=dm_constraints_obj, e8_lattice=e8_lattice_obj,
                                   time_params=time_params_dict, x_range_params=x_params_dict,
                                   lambda3_resilience_factor=0.9)

    history_data, collapse_events_data = simulator_obj.run_simulation(steps=num_iter_steps)
    final_psi_field_data = simulator_obj.Ψ # This is the final spatial field Ψ(x)

    # --- Archive Simulation Data ---
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_id = f"run_{timestamp_str}"
    data_filename = f"{run_id}_simulation_data.json"
    index_filename = "simulation_index.csv"

    # Construct packaged_results
    sim_params = {
        'lambda3_resilience': simulator_obj.λ3, # simulator.λ3 is the effective lambda3
        'Lambda_CDM_param': Λ_CDM,
        'num_simulation_steps_planned': num_iter_steps, # num_iter_steps was passed as 'steps'
        'x_range_min': simulator_obj.x_range.min(),
        'x_range_max': simulator_obj.x_range.max(),
        'x_range_points': len(simulator_obj.x_range),
        'h_sigma_param': h_σ,
        'C_scaling_param': C
    }

    packaged_results = {
        'parameters': sim_params,
        'stability_history_log': history_data, # From run_simulation
        'collapse_events_log': collapse_events_data, # From run_simulation
        'final_psi_field_real': np.real(final_psi_field_data).tolist(),
        'final_psi_field_imag': np.imag(final_psi_field_data).tolist(),
        'final_steps_executed': len(history_data) # Using len of history for actual steps
    }

    try:
        with open(data_filename, 'w') as f_json:
            json.dump(packaged_results, f_json, indent=4)
        print(f"Simulation data saved to {data_filename}")
    except IOError as e_json: # Catch specific IOError
        print(f"Error saving simulation data to JSON ({data_filename}): {e_json}")
    except TypeError as e_type: # Catch potential type errors during serialization
        print(f"TypeError during JSON serialization for {data_filename}: {e_type}")


    # Update Index File
    try:
        # These are already available from packaged_results or sim_params
        lambda_cdm_val = sim_params['Lambda_CDM_param']
        lambda3_val = sim_params['lambda3_resilience']
        sim_steps_planned_val = sim_params['num_simulation_steps_planned']
        total_steps_executed_val = len(history_data) # or packaged_results['final_steps_executed']
        stable_steps_count_val = sum(1 for log_entry in history_data if log_entry['stability'])
        collapse_events_count_val = len(collapse_events_data)

        csv_header = "RunID,Timestamp,Lambda_CDM,Lambda3_Resilience,SimStepsPlanned,TotalStepsExecuted,StableStepsCount,CollapseEventsCount,PathToDataFile"
        csv_row = f"{run_id},{timestamp_str},{lambda_cdm_val},{lambda3_val},{sim_steps_planned_val},{total_steps_executed_val},{stable_steps_count_val},{collapse_events_count_val},{data_filename}"

        file_exists_and_not_empty = os.path.exists(index_filename) and os.path.getsize(index_filename) > 0
        with open(index_filename, 'a') as f_csv:
            if not file_exists_and_not_empty:
                f_csv.write(csv_header + '\n')
            f_csv.write(csv_row + '\n')
        print(f"Simulation index updated: {index_filename}")
    except IOError as e_csv: # Catch specific IOError
        print(f"Error updating simulation index CSV ({index_filename}): {e_csv}")
    except Exception as e_generic: # Catch other potential errors
         print(f"An unexpected error occurred during CSV update: {e_generic}")
    # --- End of Archiving ---

    timestamp_suffix_plots = datetime.now().strftime("%Y%m%d_%H%M%S") # New timestamp for plot to avoid filename collision if run quickly
    plot_file = f"psi_merged_plot_{timestamp_suffix_plots}.png"
    plot_results(history_data, simulator_obj.x_range, final_psi_field_data, filename=plot_file)

    pdf_gen = SimulationReport(); pdf_gen.alias_nb_pages(); pdf_gen.add_page()
    pdf_gen.chapter_title("Simulation Setup")
    # Using sim_params for consistency in PDF report
    config_str = (
        f"Constants: h_σ={sim_params['h_sigma_param']:.4f}, C={sim_params['C_scaling_param']:.4f}, Λ_CDM={sim_params['Lambda_CDM_param']}\n"
        f"DM Params: Density={dm_constraints_obj.density_parameter}, X-Section={dm_constraints_obj.interaction_cross_section}, λ₃={dm_constraints_obj.lambda3}\n"
        f"E8 Lattice: Scale={e8_lattice_obj.attractors.max():.2f} (approx component max), Num Attractors={len(e8_lattice_obj.attractors)}\n"
        # time_params_dict is defined, use it for these specific details
        f"Time Span for ODE (conceptual): [{time_params_dict['start']},{time_params_dict['end']}] (Points for t_eval: {time_params_dict['points']})\n"
        f"Spatial Domain (x): [{sim_params['x_range_min']:.2f},{sim_params['x_range_max']:.2f}] (Points: {sim_params['x_range_points']})\n"
        f"Simulation Iteration Steps Planned: {sim_params['num_simulation_steps_planned']}\n"
        f"Effective λ₃ (Resilience Applied): {simulator_obj.λ3:.3f}"
    )
    pdf_gen.chapter_body(config_str)

    pdf_gen.chapter_title("Simulation Outcome")
    if history_data:
        last_entry = history_data[-1]
        outcome_str = (
            f"Total Steps Executed: {len(history_data)}\n"
            f"Collapse Events Triggered: {len(collapse_events_data)}\n"
            f"Final State (Step {last_entry['step']}):\n"
            f"  Stability: {'Stable' if last_entry['stability'] else 'Unstable'}\n"
            f"  Holonomy: {last_entry.get('holonomy', 'N/A'):.4f}\n"
            f"  η_E (Stress-Energy Metric): {last_entry.get('η_E', 'N/A'):.4f}\n"
            f"  ΔΘ (Phase Coherence Metric): {last_entry.get('ΔΘ', 'N/A'):.4f}\n"
            f"  Mean |Ψ(x)| (Spatial): {last_entry.get('psi_mean_abs', 'N/A'):.4f}\n"
            f"  a2_mean_abs: {last_entry.get('a2_mean_abs', 'N/A'):.3e}\n"
            f"  a2_mean_real: {last_entry.get('a2_mean_real', 'N/A'):.3e}\n"
            f"  a2_std_abs: {last_entry.get('a2_std_abs', 'N/A'):.3e}\n" # Corrected from last_log to last_entry
            f"  a2_at_resonance_real: {last_entry.get('a2_at_resonance_real', 'N/A'):.3e}\n"
            f"  a2_at_resonance_imag: {last_entry.get('a2_at_resonance_imag', 'N/A'):.3e}"
        )
    else: outcome_str = "Simulation did not produce history data."
    pdf_gen.chapter_body(outcome_str)

    pdf_gen.add_plot_image(plot_file, caption="Simulation Metrics and Final Ψ(x) Distribution")

    pdf_file_out = f"PsiCodex_Merged_Report_UserScript_{timestamp_suffix_plots}.pdf" # Use plot timestamp
    try: pdf_gen.output(pdf_file_out,'F'); print(f"PDF report: {pdf_file_out}")
    except Exception as e: print(f"PDF generation error: {e}")

    print("Psi-Codex Simulation (User Script Version) finished.")
