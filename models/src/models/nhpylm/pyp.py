import numpy as np
import random
from src.models.nhpylm.definitions import HPYLM_INITIAL_D, HPYLM_INITIAL_THETA, HPYLM_A, HPYLM_B, HPYLM_ALPHA, HPYLM_BETA

"""
Each node is essentially a Pitman-Yor process in the hierarchical Pitman-Yor language model
We use a type parameter because it can be either for characters (Char) or for words (UTF32String/UInt)
The root PYP (depth 0) contains zero context. The deeper the depth, the longer the context.
"""
class PYP():
    """
    Directly keep track of the children PYPs.
    The key in the Dict is the *additional* context to be *prepended to* the whole context represented by this PYP.
    For example, when the current node represents the 1-gram context "will", the keys might be "he" or "she", etc., leading to nodes representing the 2-gram contexts "he will", "she will" etc.
    """
    def __init__(self, context):
        self.children = {} # word/char : PYP
        self.parent:"PYP" = None # PYP
        
        # `tablegroups` is a `Dict` that groups the tables by the dish served. The key of the `Dict` is the dish, and the value of the `Dict` is a tablegroup, more specifically, an array which contains the customer count for each individual table in this table group.
        # In this model, each table serves only one dish, i.e. the draw of that word that follows the previous context ``G_u``. However, multiple tables might serve *the same dish*, i.e. a future draw might come up with the same word as a previous draw.
        # This is why we need a Vector to contain all those different tables serving this same dish (key)
        self.tablegroups = {} # word/char : List of ints
        # This keeps track of the number of total tables (not just table groups)
        self.ntables = 0
        # In the case of the C++ implementation, there is only a total number, while to get the individual number for a particular dish one will have to do some computation.
        self.ncustomers = 0
        # Useful only for CHPYLM. The number of times that the process has stopped at this Node.
        self.stop_count = 0
        # Useful only for CHPYLM. The number of times that the process has stopped at this Node.
        self.pass_count = 0       
        # The depth of this PYP node in the hierarchical structure.
        # Note that by definition the depth of a tree begins from 0
        self.depth = 0
        # Each PYP represents a particular context.
        # For the root node the context is ϵ.
        # Only the context char/word *at this level* is stored in this struct. To construct the complete context corresponding to this PYP, we'll have to trace all the way up to the root.
        # For example, a depth-2 node might store "she", while its parent, a depth-1 node, stores "will", whose parent, the root (depth-0) node, stores ϵ.
        # Then, the complete context will be the 2-gram "she will".
        self.context = context # single char/word

    @staticmethod
    def init_hyperparameters_at_depth_if_needed(depth: int, d_array: list, theta_array: list):
        if depth >= len(d_array):
            while len(d_array) <= depth:
                d_array.append(HPYLM_INITIAL_D)
            while len(theta_array) <= depth:
                theta_array.append(HPYLM_INITIAL_THETA)


    def need_to_remove_from_parent(self) -> bool:
        if self.parent == None:
            return False # If there's no parent then of course we can't remove it from the parent.
        elif len(self.children) == 0 and len(self.tablegroups) == 0:
            return True # If it has no child nor customers, then remove it.
        else:
            return False
    
    def get_num_tables_serving_dish(self, dish) -> int:     
        """
        This function explicitly returns the number of **tables** (i.e. not customers) serving a dish!
        """
        if dish in self.tablegroups:
            return len(self.tablegroups[dish])
        return 0

    def get_num_customers_for_dish(self, dish) -> int:
        if dish in self.tablegroups:
            return sum(self.tablegroups[dish])
        return 0
    
    def find_child_pyp(self, dish, generate_if_not_found: bool = False) -> "PYP":
        """
        Find the child PYP whose context is the given dish
        """
        if dish in self.children:
            return self.children[dish]
        if not generate_if_not_found:
            return None
        child = PYP(dish)
        child.parent = self
        child.depth = self.depth + 1
        self.children[dish] = child
        return child

    def add_customer_to_table(self, dish, table_index: int, G_0_or_parent_p_ws, d_array: list, theta_array: list, table_index_in_root: list) -> bool:
        """
        The second item returned in the tuple is the index of the table to which the customer is added.
        G_0_or_parent_p_ws either float or list of floats
        table_index_in_root is a list of single int number (because of the reference thing)
        """
        if dish in self.tablegroups:
            tablegroup = self.tablegroups[dish]
            tablegroup[table_index] += 1
            self.ncustomers += 1
            return True
        else:
            return self.add_customer_to_new_table(dish, G_0_or_parent_p_ws, d_array, theta_array, table_index_in_root)

    def add_customer_to_new_table(self, dish, G_0_or_parent_p_ws, d_array: list, theta_array: list, table_index_in_root: list) -> bool:
        self.extend_tablegroups(dish)
        if self.parent != None:
            success = self.parent.add_customer(dish, G_0_or_parent_p_ws, d_array, theta_array, False, table_index_in_root)
            assert success == True
        return None

    def extend_tablegroups(self, dish):
        if dish in self.tablegroups:
            self.tablegroups[dish].append(1)
        else:
            self.tablegroups[dish] = [1]
        self.ntables += 1
        self.ncustomers += 1

    def add_customer(self, dish, G_0_or_parent_p_ws, d_array: list, theta_array: list, update_beta_count: bool, index_of_table_in_root: list) -> bool:
        """
        Adds a customer eating a certain dish to a node.
        Note that this method is applicable to both the WHPYLM and the CHPYLM, thus the type parameter.
        d_array and θ_array contain the d values and θ values for each depth of the relevant HPYLM (Recall that those values are the same for one single depth.)
        """
        PYP.init_hyperparameters_at_depth_if_needed(self.depth, d_array, theta_array)
        d_u = d_array[self.depth]
        theta_u = theta_array[self.depth]
        if type(G_0_or_parent_p_ws) == float:
            if self.parent != None:
                parent_p_w = self.parent.compute_p_w(dish, G_0_or_parent_p_ws, d_array, theta_array)
            else:
                parent_p_w = G_0_or_parent_p_ws
        elif type(G_0_or_parent_p_ws) == list:
            parent_p_w = G_0_or_parent_p_ws[self.depth]
        else:
            raise ValueError("The G_0_or_parent_p_ws has type {}, but float or list expected.".format(type(G_0_or_parent_p_ws)))

        if not dish in self.tablegroups:
            self.add_customer_to_new_table(dish, G_0_or_parent_p_ws, d_array, theta_array, index_of_table_in_root)
            if update_beta_count:
                self.increment_stop_count()
            # Root PYP
            if self.depth == 0:
                index_of_table_in_root[0] = 0
            return True
        else:
            tablegroup = self.tablegroups[dish]
            sum = 0.0
            for k in range(len(tablegroup)):
                sum += max(0.0, float(tablegroup[k]) - d_u)
            t_u = self.ntables
            sum += (theta_u + d_u * t_u) * parent_p_w

            normalizer = 1.0/sum
            bernoulli = random.uniform(0, 1)
            stack = 0

            for k in range(len(tablegroup)):
                temp = float(tablegroup[k]) - d_u
                stack += max(0.0, temp) * normalizer
                if bernoulli <= stack:
                    self.add_customer_to_table(dish, k, G_0_or_parent_p_ws, d_array, theta_array, index_of_table_in_root)
                    if update_beta_count:
                        self.increment_stop_count()
                    if self.depth == 0:
                        index_of_table_in_root[0] = k
                    return True

            # If we went through the whole loop but still haven't returned, we know that we should add it to a new table.
            self.add_customer_to_new_table(dish, G_0_or_parent_p_ws, d_array, theta_array, index_of_table_in_root)

            if update_beta_count:
                self.increment_stop_count()
            # In this case, we added it to the newly created table, thus set the index as such.
            if self.depth == 0:
                index_of_table_in_root[0] = len(tablegroup) - 1
            
            return True



    def remove_customer_from_table(self, dish, table_index: int, table_index_in_root: list):
        # The tablegroup should always be found.
        tablegroup = self.tablegroups[dish]

        assert table_index < len(tablegroup)
        tablegroup[table_index] -= 1
        self.ncustomers -= 1
        assert tablegroup[table_index] >= 0
        # If there are no customers anymore at this table, we need to remove this table.
        if tablegroup[table_index] == 0:
            if self.parent != None:
                success = self.parent.remove_customer(dish, False, table_index_in_root)
                assert success == True
            del tablegroup[table_index]
            self.ntables -=1

            if len(tablegroup) == 0:
                # Will also have to delete the table from the count if we use that other system.
                self.tablegroups.pop(dish)

        return True

    def remove_customer(self, dish, update_beta_count: bool, index_of_table_in_root: list) -> bool:
        assert dish in self.tablegroups
        tablegroup = np.array(self.tablegroups[dish])
        index_to_remove = np.random.choice(np.arange(0, len(tablegroup)), p=tablegroup/tablegroup.sum())
        self.remove_customer_from_table(dish, index_to_remove, index_of_table_in_root)
        if update_beta_count:
            self.decrement_stop_count()
        if self.depth == 0:
            index_of_table_in_root[0] = index_to_remove
        return True


    def compute_p_w(self, dish, G_0: float, d_array: list, theta_array: float) -> float:
        PYP.init_hyperparameters_at_depth_if_needed(self.depth, d_array, theta_array)
        d_u = d_array[self.depth]
        theta_u = theta_array[self.depth]
        t_u = self.ntables
        c_u = self.ncustomers
        if not dish in self.tablegroups:
            coeff = (float(theta_u) + float(d_u) * float(t_u)) / (float(theta_u) + float(c_u))
            if self.parent != None:
                return self.parent.compute_p_w(dish, G_0, d_array, theta_array) * coeff
            else:
                return G_0 * coeff
        else:
            parent_p_w = G_0
            if self.parent != None:
                parent_p_w = self.parent.compute_p_w(dish, G_0, d_array, theta_array)
            c_uw = sum(self.tablegroups[dish])
            t_uw = len(self.tablegroups[dish])
            first_term = max(0.0, (float(c_uw) - float(d_u) * float (t_uw))/(float(theta_u) + float(c_u)))
            second_coeff = (float(theta_u) + float(d_u) * float(t_u)) / (float(theta_u) + float(c_u))
            return first_term + second_coeff * parent_p_w

    def compute_p_w_with_parent_p_w(self, dish, parent_p_w: float, d_array: list, theta_array: list) -> float:
        """
        Compute the possibility of the word/char `dish` being generated from this pyp (i.e. having this pyp as its context). 
        The equation is the one recorded in the original Teh 2006 paper.
        When is_parent_p_w == True, the third argument is the parent_p_w. Otherwise it's simply the G_0.
        """
        PYP.init_hyperparameters_at_depth_if_needed(self.depth, d_array, theta_array)
        d_u = d_array[self.depth]
        theta_u = theta_array[self.depth]
        t_u = self.ntables
        c_u = self.ncustomers
        if not dish in self.tablegroups:
            coeff = (float(theta_u) + float(d_u) * float(t_u)) / (float(theta_u) + float(c_u))
            return parent_p_w * coeff
        else:
            c_uw = sum(self.tablegroups[dish])
            t_uw = len(self.tablegroups[dish])
            first_term = max(0.0, (float(c_uw) - float(d_u) * float (t_uw))/(float(theta_u) + float(c_u)))
            second_coeff = (float(theta_u) + float(d_u) * float(t_u)) / (float(theta_u) + float(c_u))
            return first_term + second_coeff * parent_p_w



    # Methods specifically related to the character variant of PYP.


    def stop_probability(self, beta_stop: float, beta_pass: float, recursive: bool = True) -> float:
        p = (float(self.stop_count) + float(beta_stop))/(float(self.stop_count) + float(self.pass_count) + float(beta_stop) + float(beta_pass))
        if not recursive:
            return p
        else:
            if self.parent != None:
                p *= self.parent.pass_probability(beta_stop, beta_pass)
            return p


    def pass_probability(self, beta_stop, beta_pass, recursive: bool = True) -> float:
        p = (float(self.pass_count) + float(beta_pass))/(float(self.stop_count) + float(self.pass_count) + float(beta_stop) + float(beta_pass))
        if not recursive:
            return p
        else:
            if self.parent != None:
                p *= self.parent.pass_probability(beta_stop, beta_pass)
            return p

    def increment_stop_count(self):
        self.stop_count += 1
        if self.parent != None:
            self.parent.increment_stop_count()

    def decrement_stop_count(self):
        self.stop_count -= 1
        assert self.stop_count >= 0
        if self.parent != None:
            self.parent.decrement_stop_count()
    
    def increment_pass_count(self):
        self.pass_count += 1
        if self.parent != None:
            self.parent.increment_pass_count()
    
    def decrement_pass_count(self):
        self.pass_count -= 1
        assert self.pass_count >= 0
        if self.parent != None:
            self.parent.decrement_pass_count()

    def remove_from_parent(self) -> bool:
        if self.parent == None:
            return False
        self.parent.delete_child_node(self.context)
        return True

    def delete_child_node(self, dish):
        child = self.find_child_pyp(dish)
        if child != None:
            self.children.pop(dish)
        if len(self.children) == 0 and len(self.tablegroups) == 0:
            self.remove_from_parent()

    def get_max_depth(self, base: int) -> int:
        """
        Basically a DFS to get the maximum depth of the tree with this `pyp` as its root
        """
        max_depth = base
        for context in self.children:
            child = self.children[context]
            depth = child.get_max_depth(base+1)
            if depth > max_depth:
                max_depth = depth
        return max_depth

    def get_num_nodes(self) -> int:
        """
        A DFS to get the total number of nodes with this `pyp` as the root
        """
        count = len(self.children)
        for context in self.children:
            child = self.children[context]
            count += child.get_num_nodes()
        return count
    
    def get_num_tables(self) -> int:
        """
        The "length" of each tablegroup is exactly the "total number of tables" in that group.
        """
        count = self.ntables
        for context in self.children:
            child = self.children[context]
            count += child.get_num_tables()
        return count
    
    def get_num_customers(self) -> int:
        count = self.ncustomers
        for context in self.children:
            child = self.children[context]
            count += child.get_num_customers()
        return count

            
    def get_pass_counts(self) -> int:
        count = self.pass_count
        for context in self.children:
            child = self.children[context]
            count += child.get_pass_counts()
        return count

    def get_stop_counts(self) -> int:
        count = self.stop_count
        for context in self.children:
            child = self.children[context]
            count += child.get_stop_counts()
        return count

    # def get_all_pyps_at_depth -> probably not used anywhere

    """
    The functions below are related to hyperparameter (d, θ) sampling, based on the algorithm given in the Teh Technical Report
    There are 3 auxiliary variables defined, x_**u**, y_**u**i, z**u**wkj.
    The following methods sample them.
    """ 

    def sample_log_x_u(self, theta_u: float) -> float:
        """
        Note that only the log of x_u is used in the final sampling, expression (41) of the Teh technical report.
        Therefore our function also only ever calculates the log. Should be easily refactorable though.
        """
        if self.ncustomers >= 2:
            sample = np.random.beta(theta_u + 1.0, float(self.ncustomers - 1.0))
            sample += 1e-8 # Prevent underflow.
            return np.log(sample)
        else:
            return 0.0
    
    def sample_summed_y_ui(self, d_u: float, theta_u: float, is_one_minus: bool = False) -> float:
        if self.ntables >= 2:
            sum = 0.0
            for i in range(1, self.ntables): # both rust and julia have ntables-1
                denom = theta_u + d_u * float(i) 
                assert denom > 0
                prob = theta_u /denom
                y_ui = float(np.random.binomial(n = 1, p = prob))
                if is_one_minus:
                    sum += (1-y_ui)
                else:
                    sum += y_ui
            return sum
        else:
            return 0.0
    
    def sample_summed_one_minus_z_uwkj(self, d_u: float) -> float:
        """
        The sum is \sum_{j=1}^{c_**u**wk - 1} (1 - z_{**u**wkj}) in expression (40) of the Teh technical report.
        """
        sum = 0.0
        for dish in self.tablegroups:
            tablegroup = self.tablegroups[dish]
            # Each element in a `tablegroup` vector stores the customer count of a particular table
            for customercount in tablegroup:
                # There's also a precondition of c_uwk >= 2
                if customercount >= 2:
                    # Expression (38)
                    for j in range(1, customercount): # 1...n-1
                        assert float(j) - d_u > 0
                        prob = (float(j) - 1)/(float(j) - d_u)
                        sample = float(np.random.binomial(n = 1, p = prob))
                        sum += 1 - sample
        return sum