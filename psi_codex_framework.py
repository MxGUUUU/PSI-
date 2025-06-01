import numpy as np
import math
import matplotlib.pyplot as plt
from fpdf import FPDF
from scipy.special import airy

# Global parameters
h_σ = 0.1  # Holonomic perturbation factor (sigma_h in user's text)
CRITICAL_THRESHOLD = 1e-5
dϕ = 0.01  # Phase differential for holonomy calculation

# This is the fully_safe_text method as provided in the prompt for psi_codex_framework.py
def fully_safe_text(text: str) -> str:
    """
    Sanitizes text by replacing special characters and patterns with their
    ASCII equivalents or descriptive text.
    """
    # Handle e^{...} to exp(...) transformation first
    text = re.sub(r"e\^\{(.*?)\}", r"exp(\1)", text)

    replacements = {
        "Ψ": "Psi", "Δ": "Delta", "φ": "phi", "ϕ": "phi", # Added variant phi
        "λ": "lambda", "⊗": "x", "η": "eta",
        "ε": "epsilon", "π": "pi", "σ": "sigma", # Added sigma
        "→": "->", "′": "'", "−": "-", "√": "sqrt",
        "³": "^3", "²": "^2", "⁻¹": "^-1", "⁵": "^5", "⁴": "^4", "∫": "integral",
        "∇": "nabla", "∑": "sum", "·": ".", "≠": "!=", "≤": "<=", "≥": ">=",
        "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
        "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9"
    }
    for search_char, replace_char in replacements.items():
        text = text.replace(search_char, replace_char)
    return text

import re # Ensure re is imported for fully_safe_text

class E8_Z4_Lattice:
    """
    Represents the E8 lattice projected to Z4, relevant for Ψ-Codex encoding.
    This is a simplified conceptual placeholder.
    """
    def __init__(self, dimensions=8):
        self.dimensions = dimensions
        self.roots = np.identity(dimensions) # Simplified
        print("E8_Z4_Lattice initialized.")

    def project_to_Z4(self, vector):
        projected_vector = vector % 4 # Highly simplified
        print(f"Vector projected to Z4 (conceptual): {projected_vector}")
        return projected_vector

    def get_eigenstate(self, index):
        if 0 <= index < len(self.roots):
            eigenstate = self.roots[index]
            print(f"Retrieved eigenstate {index}: {eigenstate}")
            return eigenstate
        else:
            print(f"Eigenstate index {index} out of bounds.")
            return np.zeros(self.dimensions)

def resolve_holonomy(psi_initial, psi_perturbed, d_phi=dϕ):
    A_eff = np.sum((psi_perturbed - psi_initial) * np.conjugate(psi_initial)) / np.sum(psi_initial * np.conjugate(psi_initial))
    A_eff = np.clip(A_eff.real, -1.0, 1.0)
    delta_psi_holonomic = -1j * A_eff * psi_initial * d_phi
    psi_final_holonomic = psi_initial + delta_psi_holonomic
    norm_initial = np.linalg.norm(psi_initial)
    norm_final = np.linalg.norm(psi_final_holonomic)
    if norm_final > 1e-6 :
        psi_final_holonomic = (psi_final_holonomic / norm_final) * norm_initial
    print(f"Holonomy resolved: A_eff={A_eff:.4f}, Initial norm={norm_initial:.4f}, Final norm after holonomy={np.linalg.norm(psi_final_holonomic):.4f}")
    return psi_final_holonomic

def generate_psi_waveform(base_freq=1.0, num_points=1024, eigenstate_coeffs=None, lattice=None):
    x = np.linspace(-10, 10, num_points)
    ai, aip, bi, bip = airy(x * base_freq)
    psi_base = ai + 1j * bi
    lambda_factor = 1.0 + 0.1 * np.sin(math.pi * x / 5.0)
    psi_modulated = psi_base * lambda_factor
    if lattice and eigenstate_coeffs and len(eigenstate_coeffs) > 0:
        psi_lattice_component = np.zeros_like(psi_modulated, dtype=complex)
        num_eigenstates_to_mix = min(len(eigenstate_coeffs), lattice.dimensions)
        for i in range(num_eigenstates_to_mix):
            eigenstate_vector = lattice.get_eigenstate(i)
            psi_lattice_component += eigenstate_coeffs[i] * np.roll(psi_modulated, i * num_points // (10 * num_eigenstates_to_mix)) * np.exp(1j * i * math.pi / 4)
        if np.linalg.norm(psi_lattice_component) > 1e-6:
             psi_lattice_component = (psi_lattice_component / np.linalg.norm(psi_lattice_component)) * np.linalg.norm(psi_modulated) * 0.5
        psi_final = psi_modulated + psi_lattice_component
    else:
        psi_final = psi_modulated
    if np.linalg.norm(psi_final) > 1e-6:
        psi_final = psi_final / np.linalg.norm(psi_final)
    print(f"Ψ waveform generated. Base freq={base_freq}, Num points={num_points}. Norm={np.linalg.norm(psi_final):.4f}")
    return x, psi_final

class PsiCodexPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=15)
        print("PsiCodexPDF initialized.")

    def header(self):
        self.set_font('Arial', 'B', 12)
        title = "Ψ-Codex Holonomic Resonance Report" # Original title with special character
        # Apply fully_safe_text to the title before rendering
        safe_title = fully_safe_text(title)
        self.cell(0, 10, safe_title, 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title: str): # Added type hint for clarity
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        # Apply fully_safe_text to the title at the beginning of the method
        safe_title = fully_safe_text(title)
        self.cell(0, 6, safe_title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body_text: str): # Added type hint for clarity
        self.set_font('Times', '', 12)
        # Apply fully_safe_text to the body_text at the beginning of the method
        safe_body = fully_safe_text(body_text)
        self.multi_cell(0, 5, safe_body)
        self.ln()

    def add_plot(self, plot_path, title="Plot"):
        self.add_page()
        # The title for add_plot will be sanitized by chapter_title method
        self.chapter_title(title)
        try:
            self.image(plot_path, x=10, y=None, w=190)
            print(f"Added plot {plot_path} to PDF.")
        except Exception as e:
            print(f"Error adding plot {plot_path} to PDF: {e}")
            self.set_font('Arial', 'B', 10)
            self.set_text_color(255,0,0)
            self.multi_cell(0, 10, f"Error embedding plot: {plot_path}. Exception: {e}")
            self.set_text_color(0,0,0)
        self.ln(5)

    def add_text_section(self, title: str, text_content: str): # Added type hints
        """Adds a new page with a title and multiline text content.
           Text sanitization is handled by chapter_title and chapter_body."""
        self.add_page()
        self.chapter_title(title) # Will be sanitized by chapter_title
        self.chapter_body(text_content) # Will be sanitized by chapter_body


def plot_wavefunction(x, psi, title_suffix="Initial", filename="wavefunction.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, np.real(psi), label='Re(Ψ)')
    plt.plot(x, np.imag(psi), label='Im(Ψ)', linestyle='--')
    plt.title(f'Ψ Wavefunction (Real & Imaginary) - {title_suffix}')
    plt.xlabel('Spatial Dimension (x)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.abs(psi)**2, label='|Ψ|^2 (Probability Density)', color='r')
    plt.title(f'Probability Density |Ψ|^2 - {title_suffix}')
    plt.xlabel('Spatial Dimension (x)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Wavefunction plot saved as {filename}")
    return filename

def plot_holonomy_comparison(x, psi_initial, psi_final, title="Holonomy Effect", filename="holonomy_comparison.png"):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(x, np.real(psi_initial), label='Re(Ψ_initial)', color='blue')
    plt.plot(x, np.real(psi_final), label='Re(Ψ_final)', color='cyan', linestyle='--')
    plt.title('Real Parts Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(x, np.imag(psi_initial), label='Im(Ψ_initial)', color='red')
    plt.plot(x, np.imag(psi_final), label='Im(Ψ_final)', color='magenta', linestyle='--')
    plt.title('Imaginary Parts Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(x, np.abs(psi_initial)**2, label='|Ψ_initial|^2', color='green')
    plt.plot(x, np.abs(psi_final)**2, label='|Ψ_final|^2', color='lime', linestyle='--')
    plt.title('Probability Density Comparison')
    plt.legend()
    plt.grid(True)
    phase_initial = np.angle(psi_initial)
    phase_final = np.angle(psi_final)
    phase_diff = np.unwrap(phase_final - phase_initial)
    plt.subplot(2, 2, 4)
    plt.plot(x, phase_diff, label='Phase Diff (Ψ_final - Ψ_initial)', color='purple')
    plt.title('Phase Difference (Unwrapped)')
    plt.xlabel('Spatial Dimension (x)')
    plt.ylabel('Phase Difference (radians)')
    plt.legend()
    plt.grid(True)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()
    print(f"Holonomy comparison plot saved as {filename}")
    return filename

if __name__ == "__main__":
    print("Starting Ψ-Codex Framework Simulation...")
    lattice = E8_Z4_Lattice(dimensions=8)
    eigenstate_coeffs = [0.5, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05]
    x_coords, psi_initial = generate_psi_waveform(base_freq=0.5, num_points=2048, eigenstate_coeffs=eigenstate_coeffs, lattice=lattice)
    plot_initial_path = plot_wavefunction(x_coords, psi_initial, title_suffix="Initial", filename="psi_initial.png")
    perturbation_field = np.exp(1j * np.sin(x_coords * 0.2)**2 * math.pi/2) * 0.1
    psi_perturbed = psi_initial * perturbation_field
    psi_perturbed /= np.linalg.norm(psi_perturbed)
    print("Ψ waveform perturbed.")
    psi_final_holonomic = resolve_holonomy(psi_initial, psi_perturbed)
    plot_final_path = plot_wavefunction(x_coords, psi_final_holonomic, title_suffix="After Holonomy", filename="psi_final_holonomic.png")
    plot_comparison_path = plot_holonomy_comparison(x_coords, psi_initial, psi_final_holonomic,
                                                    title="Ψ State Evolution via Holonomy",
                                                    filename="psi_holonomy_comparison.png")
    pdf = PsiCodexPDF()
    pdf.add_page()

    intro_title = "Introduction to Ψ-Codex Simulation"
    intro_text = (
        "This report details the simulation of a Ψ-Codex waveform undergoing holonomic transformation. "
        "The Ψ-Codex framework (conceptually) leverages principles from E8 lattice structures, λ-calculus inspired "
        "modulations, and η-eigenstates for robust information encoding.\n\n"
        "Key simulation steps:\n"
        "1. Initialization of an E8_Z4 lattice model (conceptual).\n"
        "2. Generation of an initial Ψ waveform using Airy functions, modulated by λ-rules and mixed with η-eigenstates.\n"
        "3. Application of a simulated perturbation to the Ψ waveform.\n"
        "4. Resolution of the holonomy effect on the perturbed waveform, yielding the final Ψ state.\n"
        "5. Visualization of the initial, final, and comparative states.\n\n"
        f"Global parameters used: h_σ (Holonomic Perturbation Factor) = {h_σ}, "
        f"CRITICAL_THRESHOLD = {CRITICAL_THRESHOLD}, dϕ (Phase Differential) = {dϕ}.\n"
        "The plots and subsequent sections will illustrate these steps."
    )
    # Calls to add_text_section now pass raw strings, sanitization is internal
    pdf.add_text_section(intro_title, intro_text)

    # Add plots to PDF - title will be sanitized by chapter_title via add_plot
    pdf.add_plot(plot_initial_path, title="Initial Ψ Waveform")
    pdf.add_plot(plot_final_path, title="Ψ Waveform After Holonomic Transformation")
    pdf.add_plot(plot_comparison_path, title="Comparison: Initial vs. Final Ψ State")

    analysis_title = "Analysis of Holonomic Transformation"
    analysis_text = (
        "The holonomic transformation resulted in observable changes in the Ψ waveform's real and imaginary components, "
        "as well as its probability density and phase profile. The `resolve_holonomy` function models this by calculating an "
        "effective gauge potential (A_eff) from the difference between the initial and perturbed states, then applying a "
        "phase shift proportional to this potential and dϕ.\n\n"
        "The specific changes observed in the plots (e.g., shifts in peaks, changes in amplitude, phase distortions) "
        "are direct consequences of this simplified model. In a full Ψ-Codex system, these transformations would be "
        "part of a complex error correction and information retrieval mechanism.\n\n"
        "The stability of the waveform, indicated by its norm, is monitored throughout the process. "
        "The use of Airy functions as a base provides a non-trivial starting point, and the λ-modulations and "
        "η-eigenstate mixing (from the E8_Z4 lattice) add further layers of complexity, aiming for a rich state space "
        "for encoding information.\n\n"
        "Future work could involve:\n"
        "- More sophisticated models for the E8_Z4 lattice and η-eigenstates.\n"
        "- Physically grounded models for λ-modulations.\n"
        "- A more rigorous derivation of the holonomy and its effects based on underlying gauge theory principles.\n"
        "- Implementation of actual encoding and decoding algorithms using the Ψ-Codex framework."
    )
    pdf.add_text_section(analysis_title, analysis_text)

    # Neuro-Cognitive Integration Flowchart (Conceptual) - Special Handling for Monospace Font
    integration_title_text = "Neuro-Cognitive Integration Model (Conceptual Flowchart)"
    # The Mermaid diagram text from the problem description, slightly different from my memory. Using problem desc.
    integration_mermaid_text = """A[Trauma Event] -->|η_E > 1/8| B[Shadow Integration]
A -->|η_E <= 1/8| C[Reed-Solomon Correction]
B --> D[G!(-(-X)) factorial collapse]
D --> E[Airy Kernel Recomposition]
E --> F[Project onto E₈ lattice]
C --> F
F --> G[Ψ⊗η Stabilizer State]
G -->|Braid Group| H[Z₄-Symmetric Ψ(x)]
H -->|CFT: c=1/2| I[Primary Field ϕ(σ)]
I -->|Fusion Rules| J[Stable Identity]"""

    pdf.add_page() # Start a new page for this section

    # Chapter Title (using standard title font via chapter_title method)
    # chapter_title already handles fully_safe_text
    pdf.chapter_title(integration_title_text)

    # Mermaid Diagram (using Courier font)
    pdf.set_font("Courier", "", 9) # Set to Courier, size 9
    # fully_safe_text is still important for the Mermaid text itself
    processed_mermaid_text = fully_safe_text(integration_mermaid_text)
    pdf.multi_cell(0, 5, processed_mermaid_text, 0, "L") # Line height 5, align Left
    pdf.ln()

    # Font will be reset by the next call to add_text_section or its internal calls
    # to chapter_title/chapter_body which set their own fonts.

    complex_string_example = "Consider the equation e^{iπn/2} + λ₃ = Ψ_total. If η > ε, then Δ_result ≠ 0."
    # The complex_string_example will be sanitized by add_text_section (via chapter_body)
    # The title "Example of Text Sanitization" will be sanitized by add_text_section (via chapter_title)
    # The f-string for the body now also passes the raw complex_string_example,
    # as chapter_body will sanitize the whole thing.
    pdf.add_text_section("Example of Text Sanitization",
                         f"Original: {complex_string_example}\n\nSanitized: {fully_safe_text(complex_string_example)}") # Keep one explicit call here for demonstration in the PDF.

    pdf_output_filename = "Psi_Codex_Holonomic_Report.pdf"
    pdf.output(pdf_output_filename, "F")
    print(f"Ψ-Codex simulation complete. Report generated: {pdf_output_filename}")
