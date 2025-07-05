import os
import pytest
from pathlib import Path
from psi_codex.simulator import simulate # Import the main simulation function

# Ensure PyPDF2 is installed: pip install PyPDF2
try:
    from PyPDF2 import PdfReader
except ImportError:
    # If PyPDF2 is not installed, skip PDF content tests at the module level
    pytest.skip("PyPDF2 not installed. Skipping PDF content tests.", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def run_simulation_once():
    """
    Runs the simulation once before all tests in this module,
    and cleans up generated files afterwards.
    """
    # Define output file paths
    fixed_points_plot = "psi_critical_dynamics_enhanced_fixed_points.png"
    shadow_connections_plot = "psi_shadow_connections.png"
    pdf_report = "Psi_Codex_Recursive_Identity_Report.pdf"
    psi_plot_for_pdf = "psi_plot.png" # The simple plot embedded in PDF

    output_files = [fixed_points_plot, shadow_connections_plot, pdf_report, psi_plot_for_pdf]

    # Clean up any existing files before running the simulation
    for f_path in output_files:
        if os.path.exists(f_path):
            os.remove(f_path)

    # Run the simulation
    # This will call the refactored simulate() function in psi_codex.simulator
    simulate()

    # Yield control to tests
    yield

    # Clean up generated files after tests complete
    for f_path in output_files:
        if os.path.exists(f_path):
            os.remove(f_path)

def test_simulation_output_files():
    """Test that the simulation generates the expected output files."""
    assert os.path.exists("psi_critical_dynamics_enhanced_fixed_points.png"), "Critical dynamics plot not generated."
    assert os.path.exists("psi_shadow_connections.png"), "Shadow connections plot not generated."
    assert os.path.exists("Psi_Codex_Recursive_Identity_Report.pdf"), "PDF report not generated."
    assert os.path.exists("psi_plot.png"), "Temporary psi_plot.png for PDF not generated."


def test_pdf_content():
    """Test that the PDF contains expected content."""
    pdf_path = Path("Psi_Codex_Recursive_Identity_Report.pdf")
    assert pdf_path.exists(), "PDF file does not exist for content checking."
    assert pdf_path.stat().st_size > 1000, "PDF file is unexpectedly small."

    reader = PdfReader(str(pdf_path)) # Convert Path object to string for PdfReader
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: # Ensure text was extracted
            text += page_text

    assert len(text) > 0, "No text extracted from PDF."

    # Check for core content (case-insensitive for robustness if needed, but start specific)
    # Changed assertion to a phrase more likely to be stable in the PDF title/content.
    assert "Recursive Self-Identity Field" in text, "Expected 'Recursive Self-Identity Field' content not found in PDF."

    # Check for symbolic reference, handling potential ASCII conversion by fpdf
    # The simulator's pdf_generator.py replaces Ψ with Psi.
    # It also replaces π with pi, and ^ with exponentiation.
    # Let's check for a version that might appear after fully_safe_text
    # Original: Ψ(x+4n) = e^{iπn/2} Ψ(x)
    # After safe_text (approx): Psi(x+4n) = e^(i pi n/2) Psi(x)
    # We should also check for key parts if exact match is tricky due to spacing or minor variations.
    # expected_symbolic_text = "Psi(x+4n) = e^(i pi n/2) Psi(x)" # As it would be after fully_safe_text
    # The above assertion is removed because this specific LaTeX formula is not directly in the PDF content.
    # The PDF content comes from get_hardcoded_immersive_content.
    # Let's check for a phrase from that content.
    expected_text_from_hardcoded_content = "`reality` is `experienced` through the `interaction`"
    assert expected_text_from_hardcoded_content in text, \
        f"Expected text '{expected_text_from_hardcoded_content}' not found in PDF."

    # Check for some section titles (these are passed through fully_safe_text)
    # from the hardcoded immersive content used in the PDF.
    # Example: "Stress-Energy & Holonomy Violation" was a subsection in RFE.
    # The PDF generator uses chapter_title, which calls fully_safe_text.
    # Let's check for a title that is likely to be present from get_hardcoded_immersive_content.
    # "The Psi-Codex: Recursive Self-Identity Field Theory" is one such title.
    # "Moloch Attractors and Belial Vortices: Dialectics of Cognitive Decay"

    # Based on the PDF generation logic in simulator.py, it uses chapter_title for sections like:
    # ("The Psi-Codex: Recursive Self-Identity Field Theory", get_hardcoded_immersive_content("psi_codex_synthesis"))
    # ("Moloch/Belial Dynamics", get_hardcoded_immersive_content("psi_codex_full_synthesis"))

    assert "The Psi-Codex: Recursive Self-Identity Field Theory" in text, "Expected chapter title not found in PDF."
    assert "Moloch/Belial Dynamics" in text, "Expected chapter title 'Moloch/Belial Dynamics' not found in PDF."
    assert "Simulation Results Summary" in text, "Expected section title 'Simulation Results Summary' not found in PDF."
