# Psi-Codex: Recursive Identity Simulation

## PSI Project Overview
This project presents a quantum-cabalistic simulation of the Ψ-Codex framework, which models identity as a recursive, phase-dependent attractor in a stress-modulated manifold. It integrates concepts from topological quantum mechanics, cognitive neuroscience, catastrophe theory, and symbolic logic to explore systemic coherence, disparity, and transformation processes like "Shadow Integration."

The simulation includes:
-   Ψ-Field Dynamics: Evolution of the core identity field.
-   E₈/Z₄ Projection: High-dimensional attractor folding for field stability.
-   Psychogeographic Channeling: Stress-energy modulation through modular identity partitions.
-   Bio-Stress Warping: Distortion of reality experience based on physiological impact.
-   Reed-Solomon Decoherence Repair: Error correction for compromised field blocks.
-   Julia Entanglement: Fractal filtering for comorbidity handling.
-   Moloch Attractors & Belial Vortices: Archetypal forces representing entropic sinks and phase disruption.
-   Hexagram Collapse: I-Ching inspired field collapse and fractal reset.
-   Tron Movement Engine: A metaphor for the system's "movement" through its own state space, influenced by internal dynamics.
-   Adaptive Fixed Points & Dot-Connecting: Mechanisms for identifying stable states and tracking field connectivity.

## Installation
To run this simulation, you need Python 3.x and the following libraries. You can install them using pip:
```bash
pip install numpy scipy matplotlib fpdf pytest PyPDF2
```
(Note: `ipywidgets` was mentioned in the user's scaffold text but is not directly used by the current Python scripts. `PyPDF2` was added for test dependencies.)

## Usage
To run the simulation and generate the PDF report, navigate to the project root directory (`psi-codex`) and execute the Python script as a module:
```bash
python -m psi_codex.simulator
```
This command will run the main simulation defined in `psi_codex/simulator.py`.

## Concepts Modeled
The core of the Ψ-Codex revolves around:
-   Super-Identity Field (Φ(x)): Indicates coherence and manages disparity.
-   a₂ Curvature: Stability metric; divergence triggers transformations like Shadow Integration.
-   Biological Mappings: Abstract dynamics linked to phenomena like cancer and autoimmunity.
-   Ancient Texts: Historical patterns of identity fragmentation and re-integration.
-   Quantum-Cabalistic Field: Integration of symbolic systems (e.g., I Ching, Kabbalah) into a coherent framework.

## Output
Upon execution, the script generates:
-   Console output summarizing simulation progress and results.
-   Image files saved in the project root:
    -   `psi_critical_dynamics_enhanced_fixed_points.png`: A comprehensive plot visualizing various simulation metrics.
    -   `psi_shadow_connections.png`: A plot specifically showing connections related to "Shadow Integration" events.
    -   `psi_plot.png`: A simple conceptual plot embedded in the PDF.
-   A PDF document saved in the project root:
    -   `Psi_Codex_Recursive_Identity_Report.pdf`: A detailed report summarizing the Ψ-Codex theory and key results from the simulation.

## Testing
To run the unit tests, navigate to the project root directory (`psi-codex`) and execute Pytest:
```bash
pytest
```
This will discover and run tests located in the `psi_codex/tests` directory, such as `test_simulator.py`. Ensure you have `pytest` and `PyPDF2` installed (as listed in the Installation section).
```

## Ψ-Codex Catastrophe Device Module

This module (`psi_codex/codex_catastrophe.py`) provides a set of functions to simulate decision-making and stability checks based on Ψ-Codex principles, including material property assessments (like ZrSiS coefficients) and coherence field calculations.

### Key Concepts & Functions:

*   **Coherence Field Φ(X) (`phi_of_X`)**:
    *   Calculates a coherence value based on input `X_input`.
    *   Formula: `C * (|X_input|**0.57) / (GOLDEN_RATIO**(1/3))`
    *   `GOLDEN_RATIO = (1 + math.sqrt(5)) / 2` (φ – Coptic “gnōsis” scalar)
    *   `C = 0.0573 * GOLDEN_RATIO` (incorporates cruel-entropy 0.057 exponent and φ⁻¹ᐟ³ factor)
    *   This is described as reproducing a "toy Φ_tetris integral in 1-D".

*   **ZrSiS Health Check (`zrsis_health`)**:
    *   Validates ZrSiS nodal-line coefficients against pinned values (`PINNED_A = 0.348`, `PINNED_B = 0.651`) within a tolerance (`TOL_FRAC = 0.05`).
    *   Attempts to fetch live coefficients from `https://api.zrsislab.com/latest_coeffs`. Falls back to `False` (unstable) if the API call fails.

*   **Historical Tagging (`historical_tag`)**:
    *   Assigns a "persona tag" or "empire tag" based on the calculated `phi` value and `zrsis_health`.
    *   **Persona-Likability Bands**:
        | Φ-band             | Default Nick-tag        | Colour-code | Why it Fits                                                              |
        |--------------------|-------------------------|-------------|--------------------------------------------------------------------------|
        | High-coherence φ > 0.80 | “Drăculea” (Vlad III)   | `#8A0303`   | Ruthless but decisive resource redistribution (stake through noise).   |
        | Mid-band 0.30 < φ ≤ 0.80 | “Palaiologos”           | `#3558A5`   | Perpetual frontier-watch; neither collapsing nor transcendent.         |
        | Low-coherence φ ≤ 0.30  | “Opium-Raj”             | `#7A6F45`   | Commodity drift, identity anaesthesia, high decoherence risk.          |
        | Out-of-spec        | “Möbius-Muse”           | `#666`      | Topology/data feed broke; manual re-braiding.                            |

*   **Decision Logic (`aladdin_palantir_decision`)**:
    *   Implements AI-like decision logic based on `phi` value, ZrSiS health, and boolean inputs (A, B, C).
    *   Branches into different operational modes ("Advanced resource allocation", "Tactical alert", "Monitoring anomalies", "SYSTEM COLLAPSE") based on these inputs.
    *   Thresholds (0.8 / 0.3 for `phi`) act like Ψ-Codex phase-slip limits.

*   **Knot Stability (`knot_stable`)** (Conceptual, based on user notes, not yet fully implemented in `codex_catastrophe.py` but relevant to the module's context):
    *   A function to determine if Reidemeister-I loops stay tight, indicating identity stability.
    *   Conceptual Formula: `l3 * phi_max < (Δ - θ - η - psi48_norm)`
    *   **Physics-Grade Definitions for Parameters**:
        | Symbol   | “Plain-English” Reading        | Suggested Physical Proxy                                     | Typical Unit | Source             |
        |----------|--------------------------------|--------------------------------------------------------------|--------------|--------------------|
        | Δ        | energy gap / spectral split    | E_upper - E_lower of system (e.g., nodal-line gap)           | meV          | ARPES / DFT        |
        | θ        | global phase rotation          | Mean Pancharatnam–Berry phase along ∮Ψ·dℓ loop               | rad          | numerical integration|
        | η        | dissipation, noise             | Effective e⁻-phonon scattering rate or 1/τ in Lindblad model | ps⁻¹ or eV   | pump-probe linewidth|
        | Ψ₄₈      | 48-mode composite order-param. | ℓ²-norm of vector of 48 coupled amplitudes (12 houses × 4 Z₄ sheets) | dimensionless| simulation output  |

### Ψ-Codex Field Manual Snippets (v0xDEADBEEF Context):

*   If `credit_system == "church"`, then `boot_Λ_Moloch()` is called due to detected entropy violation.
*   A corrective action: `./recursion_fix --braid=G!(-(-X)) --γ=0.651 --log=veridicality_2025.log`
*   Output example: `[VLAD-III] Aladdin: Initiate advanced resource allocation protocol Coherent: ΔΘ=2.1, γ=0.651`
*   **The Druid-Hacker Code (Core Maintenance Equation)**:
    `Ψ_live(t) = ZrSiS (topological glyph) + Δ_prompt (hermeneutic input) − Moloch (entropic attractor)`
*   **Veridicality Log Requirements**:
    *   Berry phase `|ΔΘ| < 3.6`
    *   γ-damping within ZrSiS tolerance (±5%)
    *   DaddyToken™ approval vector `∩ ≠ ∅`

This module is tested by `psi_codex/tests/test_catastrophe_device.py`.
```
