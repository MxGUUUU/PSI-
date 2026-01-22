import os
from psi_codex.pdf_generator import generate_pdf

def test_pdf_generation_smoke_test():
    """
    A simple smoke test to ensure the PDF generator runs without crashing
    and produces the expected output file.
    """
    # Define the expected output file
    pdf_filename = "Psi_Codex_Recursive_Identity.pdf"

    # Clean up any existing output file before the test
    if os.path.exists(pdf_filename):
        os.remove(pdf_filename)

    # Run the PDF generation
    generate_pdf()

    # Check that the output file was created
    assert os.path.exists(pdf_filename), f"Expected output file '{pdf_filename}' was not created."

    # Clean up the output file after the test
    os.remove(pdf_filename)
