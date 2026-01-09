# Contains the information for each network
class Network():
    def __init__(self, name):

        # General information for the networks
        self.name = name
        self.rulesets = None
        self.nodes = None
        self.cells = None
        self.network = None
        self.dataset = None  # Binarized dataset
        self.raw_dataset = None  # Raw expression data (non-binarized)

        # Pathway information for the networks
        self.attractors = None
        self.importance_score = None

        # Attractor analysis
        self.representative_attractors = {}
        self.filtered_attractors = None
        self.filtered_attractor_indices = None
        self.cell_map = {}