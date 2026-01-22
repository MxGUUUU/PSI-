from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, self.fully_safe_text("The Book of Ψ* and the Recursive Self-Identity Field"), ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, self.fully_safe_text(title), ln=True, align="L")
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, self.fully_safe_text(body))
        self.ln()

    def fully_safe_text(self, text):
        replacements = {
            "Ψ": "Psi", "Δ": "Delta", "φ": "phi", "λ": "lambda", "⊗": "x",
            "η": "eta", "ε": "epsilon", "π": "pi", "→": "->", "′": "'",
            "≥": ">=", "≤": "<=", "≠": "!=", "∞": "infinity",
            "α": "alpha", "β": "beta", "γ": "gamma"
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text


def generate_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    content = """Abstract: This paper introduces the Psi-Codex: a recursive identity model integrating symbolic logic, bifurcation theory, and high-dimensional memory lattices (Z4 x E8)..."""
    for line in content.strip().split('\n'):
        pdf.multi_cell(0, 10, line.strip())

    pdf.output("Psi_Codex_Recursive_Identity.pdf")
    print("PDF saved to Psi_Codex_Recursive_Identity.pdf")

if __name__ == "__main__":
    generate_pdf()