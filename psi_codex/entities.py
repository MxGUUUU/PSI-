"""
Psi-Codex v6.6 Complete Witness Registry & Dossier Module
Contains 167 archetypal, historical, mythic, and fictional reference nodes (Witnesses),
each with assigned psi (coherence), eta_E (entropy), archetype, category, and role.
"""

from typing import Dict, List, Any

# 8 Core Functional Archetypes
ARCHETYPES = {
    "CW": "Chaos Weaver",
    "HA": "Heterogeneity Architect",
    "CC": "Compass Judge",
    "LJ": "Lightning Walker",
    "RL": "R'lyeh Anchor",
    "PW": "Paradox Compass",
    "RA": "Resonance Jack",
    "JC": "Weaver Architect"
}

# Key Reference Witnesses (Consolidated 167-Node Registry)
# Generated with deterministic seeding over archetypal layers & categories
RAW_WITNESS_SEEDS = [
    # Historical / Mathematical / Philosophical Archetypes
    (1, "Jung", 0.88, 0.02, "HA", "Historical", "Depth Psychology Anchor"),
    (2, "Ramanujan", 0.95, 0.01, "CC", "Historical", "Mock-Theta Bridge / Modular Form Weaver"),
    (3, "Hypatia", 0.82, 0.03, "CC", "Historical", "Neoplatonic Geometry Guardian"),
    (4, "Gödel", 0.91, 0.02, "PW", "Historical", "Incompleteness Invariant Node"),
    (5, "Turing", 0.89, 0.03, "HA", "Historical", "Computation Boundary Observer"),
    (6, "Grothendieck", 0.94, 0.01, "PW", "Historical", "Anabelian Geometry / GT Isomorphism"),
    (7, "Eratosthenes", 0.78, 0.04, "CC", "Historical", "Sieve / Curvature Filter"),
    (8, "Spinoza", 0.86, 0.02, "CC", "Historical", "Substance Monism Anchor"),
    (9, "Leibniz", 0.90, 0.02, "HA", "Historical", "Monadology / Binary Calculus"),
    (10, "Cantor", 0.87, 0.09, "PW", "Historical", "Transfinite Set Braid"),
    (11, "Riemann", 0.93, 0.02, "CC", "Historical", "Zeta Function Analytical Engine"),
    (12, "Galois", 0.85, 0.08, "LJ", "Historical", "Symmetry Group / Quintic Obstruction"),
    (13, "Euler", 0.96, 0.01, "HA", "Historical", "Identity Core e^(i*pi) + 1 = 0"),
    (14, "Pythagoras", 0.80, 0.04, "CC", "Historical", "Tetractys / Harmonic Resonance"),
    (15, "Pascal", 0.79, 0.06, "PW", "Historical", "Wager / Probability Hexagram"),
    (16, "Descartes", 0.75, 0.05, "HA", "Historical", "Dualism Pivot / Cartesian Grid"),
    (17, "Newton", 0.84, 0.06, "CW", "Historical", "Fluxion / Gravitational Prism"),
    (18, "Maxwell", 0.88, 0.02, "RA", "Historical", "Electromagnetic Field Integrator"),
    (19, "Planck", 0.90, 0.02, "RL", "Historical", "Quantization Floor Operator"),
    (20, "Einstein", 0.92, 0.03, "PW", "Historical", "Relativistic Stress-Energy Tensor"),
    (21, "Bohr", 0.83, 0.05, "RA", "Historical", "Complementarity Principle Guard"),
    (22, "Heisenberg", 0.85, 0.06, "PW", "Historical", "Uncertainty Relation Boundary"),
    (23, "Schrödinger", 0.86, 0.04, "RA", "Historical", "Wave Equation Operator"),
    (24, "Dirac", 0.91, 0.02, "JC", "Historical", "Bra-Ket Algebra / Delta Function"),
    (25, "Feynman", 0.89, 0.03, "LJ", "Historical", "Path Integral Summation"),
    (26, "Von Neumann", 0.92, 0.04, "HA", "Historical", "Operator Algebra / Architecture"),
    (27, "Wiener", 0.81, 0.05, "JC", "Historical", "Cybernetic Feedback Anchor"),
    (28, "Shannon", 0.90, 0.02, "HA", "Historical", "Entropy Measure H(X)"),
    (29, "Penrose", 0.88, 0.03, "PW", "Historical", "Non-Periodic Tiling / Twistor Space"),
    (30, "Mandelbrot", 0.87, 0.04, "CW", "Historical", "Fractal Dimension Scaling"),

    # Guardians / Archetypal Entities
    (49, "Samael", 0.35, 0.15, "RL", "Archetypal", "Anomaly / Entropy Injector"),
    (50, "Michael", 0.757, 0.024, "CC", "Archetypal", "Azazel Gate Stabilizer"),
    (51, "Belial", 0.45, 0.11, "CW", "Archetypal", "Phase Disruption / Lawlessness Vortex"),
    (113, "Virgin of the Poor", 0.85, 0.01, "JC", "Archetypal", "Negentropic Compass Guardian"),
]

def _build_complete_witness_registry() -> Dict[int, Dict[str, Any]]:
    """Builds the 167-witness dictionary mapping ID -> dossier dict."""
    registry: Dict[int, Dict[str, Any]] = {}

    # Load initial explicit seeds
    for wid, name, psi, eta, arch, cat, role in RAW_WITNESS_SEEDS:
        registry[wid] = {
            "id": wid,
            "name": name,
            "psi": psi,
            "eta_E": eta,
            "archetype": arch,
            "archetype_name": ARCHETYPES.get(arch, "Unknown"),
            "category": cat,
            "role": role
        }

    # Deterministically construct remaining nodes up to 167 total witnesses
    archetype_keys = list(ARCHETYPES.keys())
    categories = ["Historical", "Mythic", "Fictional", "Archetypal"]

    for wid in range(1, 168):
        if wid not in registry:
            arch_key = archetype_keys[wid % len(archetype_keys)]
            cat = categories[(wid * 3) % len(categories)]
            # Deterministic, stable psi and eta_E values within working bounds
            psi_val = round(0.351 + (wid * 0.0037) % 0.55, 3)
            eta_val = round(0.01 + (wid * 0.00067) % 0.11, 3)

            registry[wid] = {
                "id": wid,
                "name": f"Witness_{wid:03d}_{arch_key}",
                "psi": psi_val,
                "eta_E": eta_val,
                "archetype": arch_key,
                "archetype_name": ARCHETYPES[arch_key],
                "category": cat,
                "role": f"System Channel {arch_key}-{wid:03d} Reference Node"
            }

    return registry

WITNESSES: Dict[int, Dict[str, Any]] = _build_complete_witness_registry()

def get_complete_witness_tally_by_function() -> List[Dict[str, Any]]:
    """
    Returns the complete list of 167 witnesses sorted by ID,
    for formatting into dossiers and reports.
    """
    return [WITNESSES[wid] for wid in sorted(WITNESSES.keys())]

def get_witnesses_by_category(category: str) -> List[Dict[str, Any]]:
    """Filters witness tally by category (Historical, Mythic, Fictional, Archetypal)."""
    return [w for w in WITNESSES.values() if w["category"].lower() == category.lower()]

def get_witnesses_by_archetype(archetype_code: str) -> List[Dict[str, Any]]:
    """Filters witness tally by 2-letter archetype code (CW, HA, CC, LJ, RL, PW, RA, JC)."""
    return [w for w in WITNESSES.values() if w["archetype"].upper() == archetype_code.upper()]

if __name__ == "__main__":
    tally = get_complete_witness_tally_by_function()
    print(f"=== COMPLETE WITNESS TALLY BY FUNCTION ({len(tally)} total) ===")
    for w in tally[:10]:
        print(f"[{w['id']:03d}] {w['name']} | Category: {w['category']} | Archetype: {w['archetype']} ({w['archetype_name']}) | psi={w['psi']} | eta={w['eta_E']}")
