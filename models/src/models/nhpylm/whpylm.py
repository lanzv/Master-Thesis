from src.models.nhpylm.hpylm import HPYLM
from src.models.nhpylm.pyp import PYP
from src.models.nhpylm.definitions import HPYLM_INITIAL_D, HPYLM_INITIAL_THETA, HPYLM_A, HPYLM_B, HPYLM_ALPHA, HPYLM_BETA
import numpy as np

class WHYPLM(HPYLM):
    def __init__(self, order: int):
        super().__init__()
        # All the fields are "inherited" from HPYLM. Or, to put it another way, unlike CHPYLM, WHPYLM doesn't have its own new fields.
        # Root PYP which has no context"
        self.root = PYP(0)
        # Depth of the whole HPYLM
        self.depth: int = max(0.0, order - 1)
        # Base probability for 0-grams, i.e. G_0(w)
        self.G_0: float = 0.0
        # Array of discount parameters indexed by depth (maybe + 1?). Note that in a HPYLM all PYPs of the same depth share the same parameters.
        self.d_array:list = [HPYLM_INITIAL_D for _ in range(order)]
        #Array of concentration parameters indexed by depth+1. Note that in a HPYLM all PYPs of the same depth share the same parameters.
        self.θ_array:list = [HPYLM_INITIAL_THETA for _ in range(order)]

        # These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)
        # Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
        # For the sampling of discount d
        self.a_array = [HPYLM_A for _ in range(order)]
        # For the sampling of discount d
        self.b_array = [HPYLM_B for _ in range(order)]
        # For the sampling of concentration θ
        self.alpha_array = [HPYLM_ALPHA for _ in range(order)]
        # For the sampling of concentration θ
        self.beta_array =[HPYLM_BETA for _ in range(order)]

    def get_num_nodes(self) -> int:
        return self.root.get_num_nodes() + 1

    def get_num_tables(self) -> int:
        return self.root.get_num_tables() + 1

    def get_num_customers(self) -> int:
        return self.root.get_num_customers() + 1

    def get_pass_counts(self) -> int:
        return self.root.get_pass_counts() + 1

    def get_stop_counts(self) -> int:
        return self.root.get_stop_counts() + 1

    def sample_hyperparameters(self):
        max_depth = len(self.d_array) # -1
        sum_log_x_u_array = [0.0 for _ in range(max_depth + 1)]
        sum_y_ui_array = [0.0 for _ in range(max_depth + 1)]
        sum_one_minus_y_ui_array = [0.0 for _ in range(max_depth + 1)]
        sum_one_minus_z_uwkj_array = [0.0 for _ in range(max_depth + 1)]

        self.depth = 0
        self.depth = self.sum_auxiliary_variables_recursively(sum_log_x_u_array, sum_y_ui_array, sum_one_minus_y_ui_array, sum_one_minus_z_uwkj_array)
        self.init_hyperparameters_at_depth_if_needed(self.depth)

        for u in range(self.depth + 1):
            self.d_array[u] = np.random.beta(
                self.a_array[u] + sum_one_minus_y_ui_array[u],
                self.b_array[u] + sum_one_minus_z_uwkj_array[u])
            self.theta_array[u] = np.random.gamma(
                self.alpha_array[u] + sum_y_ui_array[u],
                1.0 / (self.beta_array[u] - sum_log_x_u_array[u])
            )

        excessive_length = max_depth - self.depth
        for _ in range(excessive_length): #excessive_lenght + 1
            self.d_array.pop()
            self.theta_array.pop()
            self.a_array.pop()
            self.b_array.pop()
            self.alpha_array.pop()
            self.beta_array.pop()