import numpy as np
import matplotlib.pyplot as plt
import argparse # Import argparse

# Constants
C = 0.0573
epsilon = 0.02
U = 0.9
psi_xt = 0.7
phi_t = np.pi / 3

def eta_E(phi, u, lambda_3):
 return C * (phi - u * lambda_3) ** 1.5 + epsilon

def delta_theta(x):
 return 3.6 - 7 * x**(-0.5) - (U - (psi_xt * np.cos(phi_t)))

def update_plot(x_val, lambda_3_val, integration_val): # Renamed args for clarity
 phi = 1.618 # golden ratio
 eta_e = eta_E(phi, 0.8, lambda_3_val)
 delta_t = delta_theta(x_val)

 fig, ax = plt.subplots(figsize=(8, 5))
 ax.axhline(y=eta_e, color='red', linestyle='--', label=f'η_E = {eta_e:.3f}')
 ax.axhline(y=delta_t, color='black', label=f'ΔΘ(x) = {delta_t:.3f}')
 
 if eta_e < delta_t:
  ax.set_facecolor('#d2f8d2')
  status = "🟢 Coherent"
 else:
  ax.set_facecolor('#f8d2d2')
  status = "🔴 Decoherence"
  if integration_val:
   status += " → Ψ → Ψ′ = G!(-(-X))"
 
 ax.set_title(f"Status: {status}")
 ax.set_xlabel("Recursive Depth x")
 ax.set_ylabel("Value")
 ax.legend()
 plt.grid(True)
 plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ψ-Codex: Recursive Identity Stability Simulator")
    parser.add_argument('--x', type=float, default=1.0, help='Value for x (recursive depth)')
    parser.add_argument('--lambda_3', type=float, default=1.2, help='Value for λ₃')
    parser.add_argument('--integration', action='store_true', help='Enable Shadow Integration display')
    
    args = parser.parse_args()
    
    print(f"Running simulation with values: x={args.x}, λ₃={args.lambda_3}, Integration={args.integration}")
    update_plot(args.x, args.lambda_3, args.integration)
