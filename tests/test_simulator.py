import os
from psi_codex.simulator import run_simulation

def test_simulator_smoke_test():
    """
    A simple smoke test to ensure the simulator runs without crashing
    and produces the expected output file.
    """
    # Define the expected output file
    plot_filename = "psi_critical_dynamics_enhanced_fixed_points.png"
    pdf_filename = "Psi_Codex_Recursive_Identity.pdf"

    # Clean up any existing output file before the test
    if os.path.exists(plot_filename):
        os.remove(plot_filename)
    if os.path.exists(pdf_filename):
        os.remove(pdf_filename)

    # Run the simulation
    run_simulation()

    # Check that the output file was created
    assert os.path.exists(plot_filename), f"Expected output file '{plot_filename}' was not created."
    assert os.path.exists(pdf_filename), f"Expected output file '{pdf_filename}' was not created."

    # Clean up the output file after the test
    os.remove(plot_filename)
    os.remove(pdf_filename)
