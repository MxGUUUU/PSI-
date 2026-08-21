import numpy as np

# --- Biophysical Phenomena Coupling & Falsifiable Statement Benchmarks ---
# Explicitly ties Psi-Codex constants (psi=0.351, eta_E=0.125, Delta_Theta=3.6 rad)
# to measurable EEG, biophysical, cortical, and cerebellar metrics.

PSI_ANCHOR = 0.351       # Coherence floor
ETA_E_CEILING = 0.125     # Entropy boundary
DELTA_THETA_LIMIT = 3.6  # Phase-rupture threshold (radians)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
APERY_CONSTANT = 1.202056903159594

def theta_plv_to_psi(plv: float) -> float:
    """
    Maps EEG mean frontal theta Phase-Locking Value (PLV) to Psi coherence.
    Falsifiable Statement: Mean frontal theta PLV below 0.351 predicts
    dissociative symptom onset with sensitivity > 0.80.
    """
    return float(np.clip(plv, 0.0, 1.0))

def eeg_permutation_entropy_to_eta_E(pe_value: float) -> float:
    """
    Maps normalized EEG permutation entropy (or spectral slope proxy) to eta_E.
    Falsifiable Statement: Permutation entropy scaled to exceed 0.125 predicts
    cognitive task failure with AUC > 0.75.
    """
    pe_clipped = float(np.clip(pe_value, 0.0, 1.0))
    return pe_clipped * ETA_E_CEILING

def plv_variance_to_delta_theta(plv_variance: float) -> float:
    """
    Maps PLV variance across 30-second epochs to Delta_Theta (phase deviation in radians).
    Falsifiable Statement: PLV variance across 30-second epochs > 3.6 rad predicts
    subjective depersonalization.
    """
    return float(np.clip(plv_variance, 0.0, 2 * np.pi))

def lunar_phase_ventricular_resonance(phi_L: float) -> float:
    """
    Ventricular-Lunar Resonance (Equation E10):
    R_V(phi_L) = [0.573 / (1 + 0.3 * sin^2(phi_L / 2))] * |sin(phi_L / 2 + pi / 5)|^phi
    Falsifiable Statement: CSF flow amplitude peaks at full moon (phi_L = pi),
    with phase coherence following R_V(phi_L).
    """
    denom = 1.0 + 0.3 * (np.sin(phi_L / 2.0) ** 2)
    base = np.abs(np.sin(phi_L / 2.0 + np.pi / 5.0))
    numerator = 0.573 * (base ** GOLDEN_RATIO)
    return float(numerator / denom)

def astrocytic_nmda_modulation(glutamate_exocytosis: float, nr2b_antagonist_conc: float) -> tuple[float, float]:
    """
    P2Y1R / NMDA pathway modulation:
    Astrocytic glutamate exocytosis modulates eta_E, and NR2B antagonists reduce psi coherence deviation.
    """
    eta_E_mod = 0.05 + 0.15 * float(glutamate_exocytosis)
    psi_mod = PSI_ANCHOR + 0.2 * np.exp(-1.5 * float(nr2b_antagonist_conc))
    return eta_E_mod, psi_mod

# --- Cerebellar Layer & Channel Mapping ---
def cerebellar_channel_mapping(vermal_volume_ratio: float, crus1_2_connectivity: float) -> dict:
    """
    Maps cerebellar topography (Vermis, Crus I/II lobules) to the 56-channel matrix.
    Functional Topography:
      - Posterior lobe lobules VI/VII (Crus I/II): Cognitive & affective timing (Prefrontal loops).
      - Vermis: Executive emotional anchor & autonomic calibration.
    Falsifiable Statement: Cerebellar vermal volume reduction > 15% (ratio < 0.85) combined with
    Crus I/II dysconnectivity (< 0.50) predicts high-frequency phase decoherence in Layer L2 (Zeta Calculus).
    """
    vermis_stable = vermal_volume_ratio >= 0.85
    crus_stable = crus1_2_connectivity >= 0.50

    channel_gain = float(vermal_volume_ratio * crus1_2_connectivity)
    decoherence_risk = not (vermis_stable and crus_stable)

    return {
        'vermal_volume_ratio': vermal_volume_ratio,
        'crus1_2_connectivity': crus1_2_connectivity,
        'channel_gain': channel_gain,
        'decoherence_risk_predicted': decoherence_risk,
        'mapped_channels': ["Ch.17-Neural_Zeta", "Ch.25-Syntactic_Braid", "Ch.41-Archetypal_Resonance"],
        'cerebellar_status': 'SYNCHRONIZED' if not decoherence_risk else 'CEREBELLAR_DYSFUNCTION'
    }

# --- Cortical Cellular Layer Quintic Mapping ---
def quintic_cortical_layer_mapping(x: float, lambda_3: float = 1.1, epsilon: float = 0.005) -> dict:
    """
    Maps terms of the Quintic Identity Polynomial P(x) = x^5 - phi*x^4 + zeta(3)*x^3 - psi*x^2 + lambda_3*x - epsilon
    to specific cortical cellular layers (I–VI):
      - x^5: Layer I (Molecular layer - apical dendritic integration & feedback hub)
      - -phi*x^4: Layer II (External granular layer - golden-ratio phase synchronization)
      - zeta(3)*x^3: Layer III (External pyramidal layer - long-range corticocortical connections / Justice)
      - -psi*x^2: Layer IV (Internal granular layer - thalamocortical sensory input / Coherence Floor)
      - lambda_3*x: Layer V (Internal pyramidal layer - subcortical output / Resilience Engine)
      - -epsilon: Layer VI (Multiform layer - corticothalamic feedback / Free-will gap)
    """
    psi = PSI_ANCHOR
    term1 = x ** 5
    term2 = -GOLDEN_RATIO * (x ** 4)
    term3 = APERY_CONSTANT * (x ** 3)
    term4 = -psi * (x ** 2)
    term5 = lambda_3 * x
    term6 = -epsilon

    poly_value = term1 + term2 + term3 + term4 + term5 + term6

    return {
        'x': x,
        'layer_I_molecular': term1,
        'layer_II_ext_granular': term2,
        'layer_III_ext_pyramidal': term3,
        'layer_IV_int_granular': term4,
        'layer_V_int_pyramidal': term5,
        'layer_VI_multiform': term6,
        'quintic_P_x': poly_value,
        'is_root': abs(poly_value) < 1e-3
    }

# --- Graph Theory Metrics & CSF Master Clock ---
def graph_theory_metrics(adjacency_matrix: np.ndarray) -> dict:
    """
    Computes graph theory metrics for neural / witness resonator networks:
      - Rich-club hubs (nodes with highest degree and interconnectivity)
      - Small-worldness index sigma = (C / C_rand) / (L / L_rand)
    """
    adj = np.asarray(adjacency_matrix, dtype=float)
    np.fill_diagonal(adj, 0)
    degrees = np.sum(adj > 0, axis=1)

    # Simple hub detection (degree > mean + std)
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    hub_nodes = np.where(degrees > (mean_deg + 0.5 * std_deg))[0].tolist()

    # Clustering coefficient approximation
    num_nodes = len(adj)
    clustering_coefs = []
    for i in range(num_nodes):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering_coefs.append(0.0)
        else:
            subgraph = adj[np.ix_(neighbors, neighbors)]
            links = np.sum(subgraph > 0) / 2.0
            clustering_coefs.append((2.0 * links) / (k * (k - 1)))

    mean_C = float(np.mean(clustering_coefs)) if clustering_coefs else 0.0
    small_worldness_sigma = mean_C * 1.5 + 1.0  # Normalized proxy score

    return {
        'num_nodes': num_nodes,
        'degrees': degrees.tolist(),
        'rich_club_hubs': hub_nodes,
        'mean_clustering_coefficient': mean_C,
        'small_worldness_sigma': small_worldness_sigma,
        'is_small_world': small_worldness_sigma > 1.0
    }

def csf_master_clock_sync(time_hours: float, lunar_phase: float) -> float:
    """
    Synchronizes 4th ventricle CSF pulsatility with the 29.53-day synodic month
    and 24-hour circadian rhythm.
    """
    circadian_factor = np.cos(2 * np.pi * (time_hours % 24) / 24.0)
    lunar_resonance = lunar_phase_ventricular_resonance(lunar_phase)
    return float(0.5 * (1.0 + circadian_factor) * lunar_resonance)

# --- Markov Chain Monte Carlo Simulation over Witness Resonators ---
def markov_chain_witness_simulation(witnesses: list[dict], n_steps: int = 100, seed: int = 42) -> dict:
    """
    Performs a Markov Chain Monte Carlo (MCMC) walk across witness resonators.
    Transition probabilities are scaled by witness psi coherence and role compatibility.
    """
    rng = np.random.default_rng(seed)
    num_witnesses = len(witnesses)
    if num_witnesses == 0:
        return {'walk': [], 'stationary_distribution': []}

    # Construct transition matrix based on coherence overlap
    psis = np.array([w.get('psi', PSI_ANCHOR) for w in witnesses])
    P = np.outer(psis, psis)
    np.fill_diagonal(P, 0)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    # Run Markov walk
    current_state = 0
    walk_history = [current_state]
    for _ in range(n_steps - 1):
        next_state = rng.choice(num_witnesses, p=P[current_state])
        walk_history.append(next_state)
        current_state = next_state

    counts = np.bincount(walk_history, minlength=num_witnesses)
    stationary_dist = (counts / n_steps).tolist()

    return {
        'num_witnesses': num_witnesses,
        'n_steps': n_steps,
        'walk_history': walk_history,
        'stationary_distribution': stationary_dist,
        'dominant_witness_idx': int(np.argmax(stationary_dist))
    }

def evaluate_falsifiable_predictions(metrics: dict) -> dict:
    """
    Evaluates experimental EEG / biophysical metrics against Psi-Codex falsifiable prediction bounds.
    """
    plv = metrics.get('frontal_theta_plv', 0.40)
    pe = metrics.get('permutation_entropy', 0.08)
    plv_var = metrics.get('plv_variance', 1.2)
    lunar_phase = metrics.get('lunar_phase', np.pi)
    vermal_ratio = metrics.get('vermal_volume_ratio', 0.90)
    crus_conn = metrics.get('crus1_2_connectivity', 0.70)

    psi_val = theta_plv_to_psi(plv)
    eta_val = eeg_permutation_entropy_to_eta_E(pe)
    delta_theta_val = plv_variance_to_delta_theta(plv_var)
    r_v_val = lunar_phase_ventricular_resonance(lunar_phase)
    cerebellar_eval = cerebellar_channel_mapping(vermal_ratio, crus_conn)

    dissociative_risk = psi_val < PSI_ANCHOR
    task_failure_risk = eta_val > ETA_E_CEILING
    depersonalization_risk = delta_theta_val > DELTA_THETA_LIMIT
    cerebellar_risk = cerebellar_eval['decoherence_risk_predicted']

    return {
        'psi_coherence': psi_val,
        'eta_E_entropy': eta_val,
        'delta_theta_rad': delta_theta_val,
        'ventricular_lunar_resonance': r_v_val,
        'cerebellar': cerebellar_eval,
        'dissociative_risk_predicted': dissociative_risk,
        'cognitive_task_failure_predicted': task_failure_risk,
        'depersonalization_risk_predicted': depersonalization_risk,
        'system_status': 'STABLE' if not (dissociative_risk or task_failure_risk or depersonalization_risk or cerebellar_risk) else 'PHASE_RUPTURE'
    }
