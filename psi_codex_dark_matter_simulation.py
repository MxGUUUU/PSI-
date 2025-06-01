# psi_codex_dark_matter_simulation.py
# (User provided script with spatial Psi field and run_simulation loop)
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial, airy
from scipy.integrate import solve_ivp # Not used in this version, but often is
import math
from fpdf import FPDF
import json # Added for JSON operations
import datetime # Added for timestamp generation
import os # Added for os.path.exists

# Theoretical concepts from Psi-Codex framework:
# - Resonance near n ≈ 2.56: Not explicitly modeled as a dynamic variable in this simulation.
# - Fractal shift 0! → 1/8: Standard 0! = 1 is used in factorial_mod_255. The theoretical shift is not currently applied.
# - The constant C = 0.0573 is mentioned in user theory but C = 0.0573 is used here as a scaling constant.
# - The constant G = 6.67430e-11 (Gravitational constant) is defined but not currently used in active calculations.

# --- Fundamental Constants & Parameters ---
h_σ = 1/16  # Ising model conformal weight, used in entropy_bound and for critical threshold calculations. (sigma)
C = 0.0573  # Speed of light / information propagation in Psi-field (user specified constant)
Λ_CDM = 0.3 # Cosmological Constant (Dark Energy density fraction) - conceptual placeholder. (Lambda)
G = 6.67430e-11 # Standard Gravitational Constant.

class DarkMatterConstraints:
    def __init__(self, density_parameter=0.25, interaction_cross_section=0.01,
                 lambda3_coupling=0.5, critical_eta_E=1/8, coherence_limit_dTheta=3.6):
        self.density_parameter = density_parameter # Omega_DM
        self.interaction_cross_section = interaction_cross_section # sigma_DM-Psi
        self.lambda3_coupling = lambda3_coupling # lambda_3
        self.critical_eta_E = critical_eta_E          # Critical threshold for stress-energy (eta_E)
        self.coherence_limit_dTheta = coherence_limit_dTheta # Stability limit for phase coherence (Delta_Theta)

    def get_potential_term(self, psi_amplitude_sq):
        return -self.density_parameter * self.interaction_cross_section * psi_amplitude_sq

    def check_stability(self, Psi_field_spatial, x_spatial_domain, lambda3_val):
        eta_E_conceptual = np.mean(np.abs(Psi_field_spatial)**2) * (1 + lambda3_val) / (1 + np.mean(x_spatial_domain)**2)
        phases = np.angle(Psi_field_spatial)
        delta_Theta_conceptual = np.std(np.unwrap(phases)) * (1 / (h_σ * C))
        delta_Theta_conceptual = np.abs(delta_Theta_conceptual)
        stable_eta_E = eta_E_conceptual < self.critical_eta_E
        stable_delta_Theta = delta_Theta_conceptual < self.coherence_limit_dTheta
        is_overall_stable = stable_eta_E and stable_delta_Theta
        holonomy_conceptual = np.sum(np.diff(np.unwrap(phases))) / (2 * np.pi)
        return is_overall_stable, holonomy_conceptual, eta_E_conceptual, delta_Theta_conceptual

def psi_recursive_update_rule(Psi_current, x_domain, dm_constraints, e8_attractors, step_num):
    dm_effect_factor = -dm_constraints.density_parameter * dm_constraints.interaction_cross_section
    Psi_dm_affected = Psi_current + dm_effect_factor * Psi_current * 0.1
    grad_real = np.gradient(np.real(Psi_dm_affected), x_domain, edge_order=2)
    grad_imag = np.gradient(np.imag(Psi_dm_affected), x_domain, edge_order=2)
    laplacian_real = np.gradient(grad_real, x_domain, edge_order=2)
    laplacian_imag = np.gradient(grad_imag, x_domain, edge_order=2)
    Psi_diffused = Psi_dm_affected + 0.01 * (laplacian_real + 1j * laplacian_imag)
    mean_psi = np.mean(Psi_diffused)
    Psi_constrained = Psi_diffused - 0.05 * mean_psi
    field_energy_metric = np.mean(np.abs(Psi_constrained)**2) * 100
    x_transformed = factorial_mod_255(int(np.clip(field_energy_metric, 0, 50)))
    airy_input = (x_transformed - 127.5) / 127.5 * 5.0
    ai_val, _, _, _ = airy(airy_input)
    scaling_factor = np.tanh(ai_val) * 0.1
    Psi_next_step = Psi_constrained * (1 + scaling_factor * np.sin(x_domain * (step_num +1) * 0.1))
    return Psi_next_step

class E8_Z4_Lattice_Attractors:
    def __init__(self, n_attractors=1, scale=1.0):
        print(f"E8_Z4_Lattice_Attractors (spatial context) initialized.")

def factorial_mod_255(n_val):
    n_int = int(n_val)
    if n_int < 0: return 0
    if n_int == 0:
        # Standard 0! = 1. (Theoretical 'fractal shift 0! → 1/8' is not applied here).
        return 1
    if n_int >= 17: return 0
    try: val = factorial(n_int, exact=True); return val % 255
    except OverflowError: return 0

def calculate_a2_coefficient(x_spatial_domain, psi_field_spatial, y_i_proxy_values):
    if not (x_spatial_domain.shape == psi_field_spatial.shape == y_i_proxy_values.shape):
        raise ValueError("Input arrays must have the same shape for element-wise operations.")
    psi_star = np.conjugate(psi_field_spatial)
    denominator = (x_spatial_domain**5) * psi_star * y_i_proxy_values - 1.0
    a2_field = np.full_like(denominator, np.nan, dtype=complex)
    epsilon_div_zero = 1e-12
    valid_mask = np.abs(denominator) > epsilon_div_zero
    a2_field[valid_mask] = 3.0 / denominator[valid_mask]
    return a2_field

class PsiCodexSimulator:
    def __init__(self, dm_constraints_obj, e8_lattice_obj,
                 x_range_array, initial_psi_field):
        self.dm_constraints = dm_constraints_obj
        self.e8_lattice = e8_lattice_obj
        self.x_range = x_range_array
        self.Psi = np.array(initial_psi_field, dtype=complex)
        self.history = []
        print(f"PsiCodexSimulator (Spatial) Initialized. x_range shape: {self.x_range.shape}, Psi shape: {self.Psi.shape}")

    def run_simulation(self, steps=10):
        print(f"Starting spatial simulation for {steps} steps...")
        simulation_params = {
            'lambda3_resilience': self.dm_constraints.lambda3_coupling,
            'Lambda_CDM_param': Λ_CDM,
            'num_simulation_steps_planned': steps,
            'x_range_min': self.x_range.min(),
            'x_range_max': self.x_range.max(),
            'x_range_points': len(self.x_range),
            'h_sigma_param': h_σ,
            'C_scaling_param': C
        }
        for step in range(steps):
            self.Psi = psi_recursive_update_rule(self.Psi, self.x_range, self.dm_constraints, self.e8_lattice, step)
            stability, holonomy, eta_E, delta_Theta = self.dm_constraints.check_stability(
                self.Psi, self.x_range, self.dm_constraints.lambda3_coupling
            )
            x_domain_for_a2 = self.x_range
            current_psi_field_for_a2 = self.Psi
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
                'step': step, 'stability': bool(stability), 'holonomy': holonomy, 'eta_E': eta_E, 'Delta_Theta': delta_Theta,
                'psi_mean_abs': np.mean(np.abs(self.Psi)),
                'a2_mean_abs': a2_mean_abs, 'a2_mean_real': a2_mean_real, 'a2_std_abs': a2_std_abs,
                'a2_at_resonance_real': a2_at_resonance_real, 'a2_at_resonance_imag': a2_at_resonance_imag
            }
            self.history.append(current_log_entry)
            if not stability:
                print(f"  Instability detected at step {step}. Simulation halted.")
                break
        print("Spatial simulation finished or halted.")
        packaged_results = {
            'parameters': simulation_params,
            'history_log': self.history,
            'final_psi_field_real': np.real(self.Psi).tolist(),
            'final_psi_field_imag': np.imag(self.Psi).tolist(),
            'final_steps_executed': len(self.history)
        }
        return packaged_results

def plot_spatial_simulation_results(x_domain, history_log, final_psi_field_real, final_psi_field_imag, dm_constraints_config_ref, filename_prefix="psi_dm_spatial"):
    final_psi_field = np.array(final_psi_field_real) + 1j * np.array(final_psi_field_imag)
    if not history_log: print("No history to plot."); return None, None
    steps = [log['step'] for log in history_log]
    stability_hist = [log['stability'] for log in history_log]
    holonomy_hist = [log['holonomy'] for log in history_log]
    eta_E_hist = [log['eta_E'] for log in history_log]
    delta_Theta_hist = [log['Delta_Theta'] for log in history_log]
    psi_mean_abs_hist = [log['psi_mean_abs'] for log in history_log]
    a2_mean_abs_hist = [log.get('a2_mean_abs', np.nan) for log in history_log]
    a2_mean_real_hist = [log.get('a2_mean_real', np.nan) for log in history_log]
    a2_std_abs_hist = [log.get('a2_std_abs', np.nan) for log in history_log]
    a2_res_real_hist = [log.get('a2_at_resonance_real', np.nan) for log in history_log]
    a2_res_imag_hist = [log.get('a2_at_resonance_imag', np.nan) for log in history_log]
    num_metrics = 5 + 5
    fig, axs = plt.subplots(num_metrics, 1, figsize=(12, num_metrics * 2.5), sharex=True)
    plt.suptitle(f"Psi-Codex Spatial Simulation Metrics ({filename_prefix})", fontsize=16)
    axs[0].plot(steps, psi_mean_abs_hist, 'o-', label='Mean |Psi(x)|'); axs[0].set_ylabel('Mean |Psi(x)|'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(steps, stability_hist, 'o-', label='Stability (1=Stable)', color='g'); axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(['Unstable', 'Stable']); axs[1].set_ylabel('Stability'); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(steps, holonomy_hist, 'o-', label='Holonomy H(Psi)'); axs[2].set_ylabel('Holonomy'); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(steps, eta_E_hist, 'o-', label='eta_E (Stress-Energy)'); axs[3].axhline(dm_constraints_config_ref.critical_eta_E, color='r', linestyle='--', label=f'eta_E Limit ({dm_constraints_config_ref.critical_eta_E:.3f})'); axs[3].set_ylabel('eta_E'); axs[3].legend(); axs[3].grid(True)
    axs[4].plot(steps, delta_Theta_hist, 'o-', label='Delta_Theta (Phase Coherence)'); axs[4].axhline(dm_constraints_config_ref.coherence_limit_dTheta, color='r', linestyle='--', label=f'Delta_Theta Limit ({dm_constraints_config_ref.coherence_limit_dTheta:.2f})'); axs[4].set_ylabel('Delta_Theta'); axs[4].legend(); axs[4].grid(True)
    axs[5].plot(steps, a2_mean_abs_hist, 'o-', label='Mean |a2|'); axs[5].set_ylabel('Mean |a2|'); axs[5].legend(); axs[5].grid(True); axs[5].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[6].plot(steps, a2_mean_real_hist, 'o-', label='Mean Re(a2)'); axs[6].set_ylabel('Mean Re(a2)'); axs[6].legend(); axs[6].grid(True); axs[6].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[7].plot(steps, a2_std_abs_hist, 'o-', label='Std Dev |a2|'); axs[7].set_ylabel('Std Dev |a2|'); axs[7].legend(); axs[7].grid(True); axs[7].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[8].plot(steps, a2_res_real_hist, 'o-', label='Re(a2) at x approx 2.56'); axs[8].set_ylabel('Re(a2) at Res'); axs[8].legend(); axs[8].grid(True); axs[8].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[9].plot(steps, a2_res_imag_hist, 'o-', label='Im(a2) at x approx 2.56'); axs[9].set_ylabel('Im(a2) at Res'); axs[9].legend(); axs[9].grid(True); axs[9].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[-1].set_xlabel('Simulation Step')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    metrics_plot_filename = f"{filename_prefix}_metrics_plots.png"; plt.savefig(metrics_plot_filename); plt.close(fig)
    print(f"Metrics plots saved as {metrics_plot_filename}")
    fig_psi, ax_psi = plt.subplots(figsize=(12, 7))
    ax_psi.plot(x_domain, np.real(final_psi_field), label='Re(Psi_final(x))'); ax_psi.plot(x_domain, np.imag(final_psi_field), label='Im(Psi_final(x))', linestyle='--'); ax_psi.plot(x_domain, np.abs(final_psi_field), label='|Psi_final(x)|', linestyle=':')
    ax_psi.set_title('Final Psi-Field Spatial Distribution'); ax_psi.set_xlabel('Spatial Domain (x)'); ax_psi.set_ylabel('Psi Amplitude')
    ax_psi.legend(); ax_psi.grid(True)
    plt.tight_layout(); psi_plot_filename = f"{filename_prefix}_final_psi_plot.png"; plt.savefig(psi_plot_filename); plt.close(fig_psi)
    print(f"Final Psi(x) plot saved as {psi_plot_filename}")
    return metrics_plot_filename, psi_plot_filename

class SpatialPDFReport(FPDF):
    def header(self): self.set_font('Arial','B',12); self.cell(0,10,'Psi-Codex Spatial Simulation Report',0,1,'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial','I',8); self.cell(0,10,f'Page {self.page_no()}/{{nb}}',0,0,'C')
    def chapter_title_custom(self, title_str):
        self.set_font('Arial','B',12); self.set_fill_color(220,220,255); self.cell(0,6,title_str,0,1,'L',1); self.ln(4)
    def chapter_body_custom(self, body_text_list):
        self.set_font('Times','',11)
        for item in body_text_list:
            if isinstance(item, tuple) and len(item) == 2:
                self.set_font('Times', item[0], 11); self.multi_cell(0,5,item[1]); self.set_font('Times', '', 11)
            else: self.multi_cell(0,5,item)
            self.ln(1)
        self.ln(2)
    def add_section(self, title, content_list):
        self.add_page(); self.chapter_title_custom(title); self.chapter_body_custom(content_list)
    def add_full_width_plot(self, plot_path, caption=""):
        self.add_page(); self.chapter_title_custom(caption if caption else "Plot")
        try: page_width = self.w - 2*self.l_margin; self.image(plot_path, x=None, y=None, w=page_width*0.95)
        except Exception as e: self.set_font('Times','B',10); self.set_text_color(255,0,0); self.multi_cell(0,10,f"Error embedding plot '{plot_path}': {e}"); self.set_text_color(0,0,0)
        self.ln(2)
    def add_fixed_points_dynamics_section(self, packaged_results):
        self.add_page(); self.chapter_title_custom("Fixed Points and Critical Dynamics")
        content = ["This simulation explores system behavior around critical values and potential fixed points, primarily through the Psi-Codex theoretical lens."]
        history = packaged_results.get('history_log', [])
        if history:
            last_log = history[-1]
            content.append(("", "The `a2` coefficient, defined conceptually as `a2 = 3 / (x^5 * psi* . y(i) - 1)`, is theorized as a recursive attractor for the gnostic identity `gnostic(iI)* = a2 * x^5 * psi*`."))
            a2_mean_abs_last = last_log.get('a2_mean_abs', 'N/A'); a2_mean_real_last = last_log.get('a2_mean_real', 'N/A'); a2_res_real_last = last_log.get('a2_at_resonance_real', 'N/A'); a2_res_imag_last = last_log.get('a2_at_resonance_imag', 'N/A')
            content.append(f"Summary of `a2` behavior from the final simulation step:")
            content.append(f"  - Mean |a2|: {a2_mean_abs_last:.3e}" if isinstance(a2_mean_abs_last, float) else f"  - Mean |a2|: {a2_mean_abs_last}")
            content.append(f"  - Mean Re(a2): {a2_mean_real_last:.3e}" if isinstance(a2_mean_real_last, float) else f"  - Mean Re(a2): {a2_mean_real_last}")
            content.append(f"  - Re(a2) near x=2.56: {a2_res_real_last:.3e}" if isinstance(a2_res_real_last, float) else f"  - Re(a2) near x=2.56: {a2_res_real_last}")
            content.append(f"  - Im(a2) near x=2.56: {a2_res_imag_last:.3e}" if isinstance(a2_res_imag_last, float) else f"  - Im(a2) near x=2.56: {a2_res_imag_last}")
            all_a2_mean_abs = [log.get('a2_mean_abs', np.nan) for log in history if isinstance(log.get('a2_mean_abs'), (int, float))]
            all_a2_mean_abs_numeric = [x for x in all_a2_mean_abs if not np.isnan(x)]
            if all_a2_mean_abs_numeric: overall_mean_a2_abs = np.mean(all_a2_mean_abs_numeric); content.append(f"The overall average of Mean |a2| across all simulation steps was: {overall_mean_a2_abs:.3e}.")
            else: content.append("Overall average of Mean |a2| could not be computed (no valid data).")
        else: content.append("No simulation history available to summarize `a2` coefficient behavior.")
        content.append(("", "Critical Thresholds (eta_E, Delta_Theta):"))
        content.append("The simulation monitors key stability indicators: stress-energy `eta_E` and phase coherence `Delta_Theta`.")
        content.append(f"Stability is assessed against thresholds: `eta_E < 1/8` ({1/8:.3f}) and `Delta_Theta < 3.6`. Breaching these can indicate departure from stable operational regimes, conceptually linked to events like 'shadow integration' in the broader theory.")
        content.append("The 'Simulation Metrics' plots visually track `eta_E` and `Delta_Theta` relative to these critical limits.")
        content.append(("", "Holonomy:"))
        content.append("Holonomy (`oint dM Psi.dl`) is another critical indicator. The simulation aims for a near-zero holonomy (`abs(holonomy) < 1e-5` as a typical goal) as a condition for topological stability. Convergence towards this state can be seen as the system seeking a topological fixed point.")
        content.append("The 'Holonomy H(Psi)' plot in the 'Simulation Metrics' section shows its behavior over the simulation steps.")
        content.append(("", "Other Theoretical Points:"))
        content.append("Other theoretical concepts like resonance near `n approx 2.56` (where `a2` was specifically analyzed) and the fractal shift `0! -> 1/8` are part of the Psi-Codex framework. While `a2` was analyzed at `x approx 2.56`, these concepts primarily provide context for interpreting critical behaviors rather than being dynamic variables in the current simulation's core equations.")
        self.chapter_body_custom(content)

if __name__ == "__main__":
    print("Initiating Psi-Codex Spatial Simulation...")
    dm_constraints_config = DarkMatterConstraints(density_parameter=0.22, interaction_cross_section=0.012, lambda3_coupling=0.55, critical_eta_E=1/8, coherence_limit_dTheta=3.6)
    e8_lattice_sim_obj = E8_Z4_Lattice_Attractors()
    x_pts = 500; x_domain_array = np.linspace(-5, 5, x_pts)
    amplitude = 1.0; width = 2.0; wavevector = 1.5
    initial_psi_spatial = amplitude * np.exp(-x_domain_array**2 / width**2) * np.exp(1j * wavevector * x_domain_array)
    initial_psi_spatial[x_pts//2:] *= 0.8
    psi_simulator_spatial = PsiCodexSimulator(dm_constraints_obj=dm_constraints_config, e8_lattice_obj=e8_lattice_sim_obj, x_range_array=x_domain_array, initial_psi_field=initial_psi_spatial)
    num_simulation_steps = 100
    packaged_simulation_results = psi_simulator_spatial.run_simulation(steps=num_simulation_steps)
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_id = f"run_{timestamp_str}"; data_filename = f"{run_id}_simulation_data.json"; index_filename = "simulation_index.csv"
    try:
        with open(data_filename, 'w') as f_json: json.dump(packaged_simulation_results, f_json, indent=4)
        print(f"Simulation data saved to {data_filename}")
    except Exception as e_json: print(f"Error saving simulation data to JSON: {e_json}")
    try:
        params = packaged_simulation_results['parameters']; history = packaged_simulation_results['history_log']
        lambda_cdm = params['Lambda_CDM_param']; lambda3 = params['lambda3_resilience']; sim_steps_planned = params['num_simulation_steps_planned']
        total_steps_executed = packaged_simulation_results['final_steps_executed']
        stable_steps_count = sum(1 for log_entry in history if log_entry['stability']) # Corrected key to 'stability'
        csv_header = "RunID,Timestamp,Lambda_CDM,Lambda3_Resilience,SimStepsPlanned,TotalStepsExecuted,StableStepsCount,PathToDataFile\n"
        csv_row = f"{run_id},{timestamp_str},{lambda_cdm},{lambda3},{sim_steps_planned},{total_steps_executed},{stable_steps_count},{data_filename}\n"
        file_exists = os.path.exists(index_filename)
        with open(index_filename, 'a') as f_csv:
            if not file_exists or os.path.getsize(index_filename) == 0: f_csv.write(csv_header)
            f_csv.write(csv_row)
        print(f"Simulation index updated: {index_filename}")
    except Exception as e_csv: print(f"Error updating simulation index CSV: {e_csv}")
    simulation_history = packaged_simulation_results['history_log']
    final_psi_field_real = packaged_simulation_results['final_psi_field_real']
    final_psi_field_imag = packaged_simulation_results['final_psi_field_imag']
    metrics_plot_file, psi_plot_file = plot_spatial_simulation_results(x_domain_array, simulation_history, final_psi_field_real, final_psi_field_imag, dm_constraints_config, filename_prefix="psi_dm_spatial_sim")
    pdf = SpatialPDFReport(); pdf.alias_nb_pages()
    params_report = packaged_simulation_results['parameters']
    config_summary = [f"h_sigma: {params_report['h_sigma_param']}", f"C (Scaling Constant): {params_report['C_scaling_param']}", f"Lambda_CDM: {params_report['Lambda_CDM_param']}", f"Spatial Domain (x): min={params_report['x_range_min']:.2f}, max={params_report['x_range_max']:.2f}, points={params_report['x_range_points']}", f"Num Sim Steps Planned: {params_report['num_simulation_steps_planned']}", f"Num Sim Steps Executed: {packaged_simulation_results['final_steps_executed']}", f"DM Density (Omega_DM): {dm_constraints_config.density_parameter}",  f"DM Interaction (sigma_DM-Psi): {dm_constraints_config.interaction_cross_section}", f"lambda_3 Coupling (Resilience): {params_report['lambda3_resilience']}", f"eta_E Crit Threshold: {dm_constraints_config.critical_eta_E:.4f}", f"Delta_Theta Coherence Limit: {dm_constraints_config.coherence_limit_dTheta:.2f}",]
    pdf.add_section("Simulation Configuration", config_summary)
    if simulation_history:
        final_log = simulation_history[-1]
        results_summary = [f"Final Step: {final_log['step']}", f"Stability: {'Stable' if final_log['stability'] else 'Unstable'}", f"Holonomy H(Psi): {final_log['holonomy']:.4f}", f"eta_E: {final_log['eta_E']:.4f}", f"Delta_Theta: {final_log['Delta_Theta']:.4f}", f"Mean |Psi(x)|: {final_log['psi_mean_abs']:.4f}", f"Mean |a2|: {final_log.get('a2_mean_abs', 'N/A'):.3e}", f"Mean Re(a2): {final_log.get('a2_mean_real', 'N/A'):.3e}", f"Std Dev |a2|: {final_log.get('a2_std_abs', 'N/A'):.3e}", f"Re(a2) at x approx 2.56: {final_log.get('a2_at_resonance_real', 'N/A'):.3e}", f"Im(a2) at x approx 2.56: {final_log.get('a2_at_resonance_imag', 'N/A'):.3e}",]
        pdf.add_section("Final State Summary & Conceptual Metrics", results_summary)
    else: pdf.add_section("Simulation Results", ["Simulation did not produce history."])
    pdf.add_fixed_points_dynamics_section(packaged_simulation_results)
    if metrics_plot_file: pdf.add_full_width_plot(metrics_plot_file, caption="Time Evolution of Simulation Metrics")
    if psi_plot_file: pdf.add_full_width_plot(psi_plot_file, caption="Final Psi-Field Spatial Distribution")
    model_desc = ["This simulation models Psi(x) evolution via recursive updates.", "Each step involves conceptual DM interaction, E8_Z4 constraint, and factorial/Airy kernel transformation.", "Stability (eta_E, Delta_Theta) and holonomy are assessed. a2 = 3 / (x^5*psi* . y(i) - 1) (proxy y(i)=sin(x^2+3)) stats are logged.", "Model is conceptual for exploring Psi-Codex theory."]
    pdf.add_section("Model Description", model_desc)
    pdf_filename = "Psi_Codex_Spatial_DM_Report.pdf"
    try: pdf.output(pdf_filename, "F"); print(f"Spatial simulation PDF report generated: {pdf_filename}")
    except Exception as e: print(f"Error generating PDF: {e}")
    print("Psi-Codex Spatial Simulation finished.")
