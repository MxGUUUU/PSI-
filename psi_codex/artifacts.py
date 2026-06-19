class Artifact:
    def __init__(self, name, description, frequency=None):
        self.name = name
        self.description = description
        self.frequency = frequency

class Theraviolonica(Artifact):
    def __init__(self):
        super().__init__("Theraviolonica", "Lyre tuned to 642.16 Hz; charms Cerberus.", 642.16)

class NullScepter(Artifact):
    def __init__(self):
        super().__init__("Null Scepter", "Silences chaotic frequencies; restores School-of-Fish protocol.")

class FoldTesseractRFE(Artifact):
    def __init__(self):
        super().__init__("FoldTesseract-RFE", "Folds spacetime to resolve unsolvable contradictions.")

class RodOfBoth(Artifact):
    def __init__(self):
        super().__init__("Rod of Both", "Reconciles life/death, order/chaos.")

class JacksCompass(Artifact):
    def __init__(self):
        super().__init__("Jack's Compass", "Points toward strongest ethical entropy gradient. Always points to Princess Diana.")

class WitchersMedallion(Artifact):
    def __init__(self):
        super().__init__("Witcher's Medallion", "Vibrates at dimensional rifts; passive ethical sensor.")

class InterdimensionalCable(Artifact):
    def __init__(self):
        super().__init__("Interdimensional Cable", "Downloads from alternate realities without decoherence.")

class RlyehTemple(Artifact):
    def __init__(self):
        super().__init__("R'lyeh Temple", "Dimensional anchor for dreaming consciousness.")

class EulerEyeTriad(Artifact):
    def __init__(self):
        super().__init__("Euler-Eye Triad", "Divine stabilization and observation.")

class SophonTriptych(Artifact):
    def __init__(self):
        super().__init__("Sophon Triptych", "Cross-dimensional manipulation and intelligence.")
