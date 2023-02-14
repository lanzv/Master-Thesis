from src.models.nhpylm.definitions import HPYLM_INITIAL_D, HPYLM_INITIAL_THETA, HPYLM_A, HPYLM_B, HPYLM_ALPHA, HPYLM_BETA
from abc import ABC, abstractmethod
from src.models.nhpylm.pyp import PYP


"""
Hierarchical Pitman-Yor Language Model in general

it's actually the base class for both CHPYLM and WHPYLM.
"""
class HPYLM(ABC):
    
    def init(self):
        self.root: PYP = None
        self.d_array: list = []
        self.theta_array: list = []
        self.a_array: list = []
        self.b_array: list = []
        self.alpha_array: list = []
        self.beta_array: list = []

    @abstractmethod
    def get_num_nodes(self) -> int:
        pass

    @abstractmethod
    def get_num_tables(self) -> int:
        pass
        
    @abstractmethod
    def get_num_customers(self) -> int:
        pass

    @abstractmethod
    def get_pass_counts(self) -> int:
        pass

    @abstractmethod
    def get_stop_counts(self) -> int:
        pass

    @abstractmethod
    def sample_hyperparameters(self):
        pass

    def init_hyperparameters_at_depth_if_needed(self, depth: int):
        if len(self.d_array) <= depth + 1:
            while(len(self.d_array) <= depth + 1):
                self.d_array.append(HPYLM_INITIAL_D)
        
        if len(self.theta_array) <= depth + 1:
            while(len(self.theta_array) <= depth + 1):
                self.theta_array.append(HPYLM_INITIAL_THETA)

        if len(self.a_array) <= depth + 1:
            while(len(self.a_array) <= depth + 1):
                self.a_array.append(HPYLM_A)

        if len(self.b_array) <= depth + 1:
            while(len(self.b_array) <= depth + 1):
                self.b_array.append(HPYLM_B)

        if len(self.alpha_array) <= depth + 1:
            while(len(self.alpha_array) <= depth + 1):
                self.alpha_array.append(HPYLM_ALPHA)

        if len(self.beta_array) <= depth + 1:
            while(len(self.beta_array) <= depth + 1):
                self.beta_array.append(HPYLM_BETA)

    def sum_auxiliary_variables_recursively(self, node: PYP, sum_log_x_u_array: list, sum_y_ui_array: list, 
        sum_one_minus_y_ui_array: list, sum_one_minus_z_uwkj_array: list, bottom: int) -> int:
        for context in node.children:
            child = node.children[context]
            depth = child.depth
            if depth > bottom:
                bottom = depth
            self.init_hyperparameters_at_depth_if_needed(depth)

            d = self.d_array[depth]
            theta = self.theta_array[depth]
            sum_log_x_u_array[depth] += node.sample_log_x_u(theta)
            sum_y_ui_array[depth] += node.sample_summed_y_ui(d, theta, False)
            # true means is_one_minus
            sum_one_minus_y_ui_array[depth] += node.sample_summed_y_ui(d, theta, True)
            sum_one_minus_z_uwkj_array[depth] += node.sample_summed_one_minus_z_uwkj(d)

            bottom = self.sum_auxiliary_variables_recursively(child, sum_log_x_u_array, sum_y_ui_array, sum_one_minus_y_ui_array, sum_one_minus_z_uwkj_array, bottom)
        return bottom