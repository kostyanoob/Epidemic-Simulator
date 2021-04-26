import pulp as pl
import numpy as np
from Institution import Institution


class SelectCandidatesForTest:
    """
    Linear Programming problem. Receives the institution (which contains its organizational structure) and allowed
    number of tests. The goal o the problem is to sample the optimal candidates out of the people, such that
    the coverage of the overall risk in the groups is maximized.

    """
    def __init__(self, test_budget: int, institution: Institution,
                 target_people_ids: list,
                 test_overview,
                 integer_programming=False,
                 normalized_coverage=True,
                 secondary_objective_coefficient=0.01):
        """

        :param test_budget: maximum number of allowed tests (budget)
        :param institution: an object encompassing the structure of the organization
        :param target_people_ids: an id list of people that will be subject to selection.
        :param integer_programming: True --> the linear program becomes an integer program.
                                    False --> the linear program is solved fractionally, then randomly rounded.
        :param normalized_coverage: Sets the type of coverage used in the constraints.
                                    False --> coverage(group) = sum of weights of tested people in the group
                                    True  --> coverage(group) = sum of weights of tested people in the group / weight of the group
        :param secondary_objective_coefficient: float - a coefficient to multiply the secondary objective of the optimization.
        """

        if len(institution.person_lst) < test_budget:
            self.B = len(institution.person_lst)  # number of allotted tests
            print("Warning: The budget of B={} cannot be exploited since there are only "
                  "{} people in the organization. The budget was therefore "
                  "truncated to {}".format(test_budget, self.B, self.B))
        else:
            self.B = test_budget

        self.institution = institution
        self.integer_programming = integer_programming
        self.problem = pl.LpProblem("Institution_People_Sampling_for_CoVID-19_Testing", sense=pl.LpMaximize)
        self.z = pl.LpVariable("z", cat=pl.LpContinuous)
        self.x = pl.LpVariable.dicts("x", list(target_people_ids), lowBound=0.0, upBound=1.0, cat=pl.LpBinary if integer_programming else pl.LpContinuous)

        person_weight = {}
        for personid in target_people_ids:
            person_weight[personid] = self.institution.speculative_person_weight(personid, "Optimization", test_overview)

        # Compute group coverages c(e) = <x,w>/W
        group_coverage = {}
        all_people_weight = sum(person_weight.values())
        for group_idx, group in enumerate(institution.group_lst):
            group_people_id_lst = list(set(institution.G.neighbors(group)).intersection(target_people_ids))
            group_people_var_lst = [self.x[personid] for personid in group_people_id_lst]
            group_weight = sum(person_weight[personid] for personid in group_people_id_lst)
            if normalized_coverage:
                group_people_weight_lst = [person_weight[personid] / group_weight if group_weight != 0 else 0.0 for personid in group_people_id_lst]
            else:
                group_people_weight_lst = [person_weight[personid] for personid in group_people_id_lst]
            group_coverage[group] = pl.lpDot(group_people_var_lst, group_people_weight_lst)

            # FOR EVERY GROUP set a constraint c(e) <= z
            if group_weight >= (0.5 / len(target_people_ids)) * all_people_weight: # must avoid constraining on the non-risky groups
                self.problem += group_coverage[group] >= self.z, "group_{}_coverage".format(group_idx)

        # Sum of the sampled person must not exceed the number of allotted tests (B)
        self.problem += pl.lpSum(self.x[pid] for pid in target_people_ids) <= self.B, "Constraint_on_the_maximum_number_of_tests"

        # Primary objective - fairness; Secondary objective - sum of coverages
        regularizer = secondary_objective_coefficient / len(institution.group_lst) if len(institution.group_lst) != 0 else 0.1
        self.problem += self.z + regularizer * pl.lpSum(group_coverage.values())

    def __str__(self):
        return "People Selection Linear Program with the " \
               "following parameters:\n B={}:\n{}".format(self.B, self.problem)

    def solve(self, path=None, verbosity=0):
        """
        Solves the LP problem and returns the list of people chosen for sampling.
        :param path: str - path to the solver executable file
        :param verbosity: 0 - no messages, 1 - only python messages, 2 - python and solver messages
        :return: list of person chosen for sampling, or None if failed
        """
        try:
            self.problem.solve(pl.GLPK_CMD(msg=verbosity > 1, path= path))
            solver_crashed = False
        except:
            solver_crashed = True

        if not solver_crashed and self.problem.status == 1:
            sampled_person_lst = []
            person_idx_to_x_value_dict = {person_id:pl.value(x) for person_id, x in sorted(self.x.items())}
            for person_idx, x_value in sorted(person_idx_to_x_value_dict.items()):
                if self.integer_programming:
                    sampled = x_value
                    if verbosity > 0:
                        print("person {} : {}".format(person_idx, pl.value(x_value)))
                else:  # Requires randomized rounding
                    sampled = np.random.binomial(1, pl.value(x_value))
                    if verbosity > 0:
                        print("person {:25} : {} randomly rounded to {} ".format(person_idx, x_value, sampled))
                if sampled:
                    sampled_person_lst.append(person_idx)

            # if the number of sampled people is different than B - add/remove people as required to reach exactly B:
            sampled_person_lst = self.refine_sampled_person_lst(sampled_person_lst, person_idx_to_x_value_dict,
                                                                verbosity=verbosity)

            # Report the sampled people
            num_selected_people = len(sampled_person_lst)
            if verbosity > 0:
                print("Found a solution (z={}): People chosen for sampling".format(pl.value(self.z)))
                print(sampled_person_lst)
                print("-" * 72)
                print("The solution selected {} {} (B was set to {})".format(num_selected_people, "person" if num_selected_people == 1 else "people", self.B))
            return sampled_person_lst
        else:
            if verbosity > 0:
                print("Failed solving the linear program of people selection")
            return None

    def status(self):
        return self.problem.status

    def refine_sampled_person_lst(self, sampled_person_lst : list, person_idx_to_x_value_dict : dict, verbosity : int):
        """
        If, as a result of the linear problem solution, the number of sampled people is not exactly B,
        this function will attempt to bring it as close to B as possible.
        :param sampled_person_lst: list (of ints) of people that were chosen for testing by the solver.
        :param person_idx_to_x_value_dict: a dictionary where person indices are the keys and the values are the
                                       optimization variables.
        :param verbosity: integer, any positive value will allow messages to be printed to the stdout.
        :return: a refined list of people chosen for testing.
        """
        num_selected_people = len(sampled_person_lst)

        # Phase 1 - try to correct the selection list, by adding or removing people from the unselected non-isolated and
        #           non-symptomatic people
        if not self.integer_programming and num_selected_people != self.B:
            if verbosity > 0: print("Number of chosen people {} doesn't "
                                    "correspond to budget B={}. Starting "
                                    "refinements...".format(num_selected_people, self.B))
            if num_selected_people > self.B:
                sampled_person_lst_ordered_by_score = [person_idx for person_idx, value
                                                       in sorted(person_idx_to_x_value_dict.items(), key=lambda item: item[1])
                                                       if person_idx in sampled_person_lst]
                while len(sampled_person_lst_ordered_by_score) > self.B:  # continue removing people with ascending score
                    person = sampled_person_lst_ordered_by_score.pop(0)
                    if verbosity > 0:
                        print(" - Removing person named {}".format(person))
                sampled_person_lst = sampled_person_lst_ordered_by_score
            elif num_selected_people < self.B:
                unsampled_person_lst_ordered_by_score = [person_idx for person_idx, value
                                                         in sorted(person_idx_to_x_value_dict.items(), key=lambda item: item[1])
                                                         if person_idx not in sampled_person_lst]
                while len(sampled_person_lst) < self.B and len(unsampled_person_lst_ordered_by_score) > 0:
                    person = unsampled_person_lst_ordered_by_score.pop(-1)
                    sampled_person_lst.append(person)
                    if verbosity > 0:
                        print(" - Adding person named {}".format(person))

        # Phase 2 - The number of selected people may still not reach the budget B. This can happen if a major portion
        # of the organization is isolated already. In this case, spend some tests on isolated people which haven't been
        # tested for long enough
        id_isolated_people_list = filter(lambda i: not self.institution.nodes_attributes[i]['isolated'][0], self.institution.id_lst)
        isolated_people_sorted_by_increasing_isolation_period = list(sorted(id_isolated_people_list, key=lambda i: self.institution.nodes_attributes[i]['isolated'][1]))
        while len(sampled_person_lst) < self.B and len(isolated_people_sorted_by_increasing_isolation_period) > 0:
            isolated_person_for_ahead_of_line_check = isolated_people_sorted_by_increasing_isolation_period.pop()
            if verbosity > 0:
                print(" - Adding person id {}".format(isolated_person_for_ahead_of_line_check))
            sampled_person_lst.append(isolated_person_for_ahead_of_line_check)

        return sampled_person_lst




