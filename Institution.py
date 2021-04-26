from typing import Tuple, List, Union, Iterable
from RiskManager import RiskManager
from MyDate import MyDate
from Util.numeric import modified_sigmoid_vector
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import ceil
__author__ = "Kostya Berestizshevsky"
__version__ = "0.1.0"
__license__ = "MIT"
import os
import itertools
import numpy as np
import networkx as nx
import pandas
import matplotlib
from typing import List, Tuple
matplotlib.use("Agg")


class Institution:

    def __init__(self, organization_df: pandas.DataFrame, ext_risk_df: pandas.DataFrame,
                 group_risk_df: pandas.DataFrame, int_risk_df: pandas.DataFrame, delta_df: pandas.DataFrame,
                 current_date: MyDate, risk_manager: RiskManager):
        # Flat Data
        self.num_organization_columns_that_arent_group_names = 5
        self.num_ext_risk_df_columns_that_arent_risk_factors = 5
        self.num_int_risk_df_columns_that_arent_risk_factors = 3
        self.num_group_risk_df_columns_that_arent_risk_factors = 2

        self.log = organization_df[organization_df.columns[:3]].copy()
        self.isol_log = organization_df[organization_df.columns[:3]].copy()
        person_details_columns_list = []
        for i in range(self.num_organization_columns_that_arent_group_names-1):
            person_details_columns_list.append(
                organization_df[organization_df.columns[i]])
        self.current_date = current_date
        self.group_lst = list(
            organization_df.columns[self.num_organization_columns_that_arent_group_names:])
        self.person_lst = ["_".join(wid_cid_fn_pos_Tuple) for wid_cid_fn_pos_Tuple in zip(
            *person_details_columns_list)]
        self.person_idx_to_name_dict = {
            pid: pname for pid, pname in enumerate(self.person_lst)}
        self.group_idx_to_name_dict = {
            gid: gname for gid, gname in enumerate(self.group_lst)}
        self.person_name_to_idx_dict = {
            pname: pid for pid, pname in enumerate(self.person_lst)}
        self.group_name_to_idx_dict = {
            gname: gid for gid, gname in enumerate(self.group_lst)}
        self.id_lst = list(self.person_idx_to_name_dict.keys())
        # infected, ext_infection,int_infection
        self.infection_stat = [0, 0, 0]
        self.day_peak_max = (0, '')
        self.day_total_ill_peak_max = (0, '')

        # Graph
        self.G = nx.Graph()
        self.G.add_nodes_from(self.id_lst + self.group_lst)
        for personid, group in itertools.product(self.id_lst, self.group_lst):
            if organization_df[group][personid]:
                self.G.add_edge(personid, group)

        # Set the weight attribute for each node in the graph - both in self.nodes_attributes and in self.G
        self.risk_manager = risk_manager
        self.nodes_attributes = {}
        self.init_nodes_attributes(
            organization_df, ext_risk_df, delta_df, group_risk_df, int_risk_df)

    def get_discount_f_neg(self, t: MyDate, ts: MyDate):
        """
        Given the current time "t" and the most recent time ts at which a negative-resulting test was made
        return the f_neg(time_interval) function, which is one of the terms of the discount

        :param t  current time step
        :param ts: most recent time of sampling (disease testing)
        :return: float in [0,1], the risk discount factor that should be applied to the weighted risks.
        """
        if ts is None:
            return 1.0
        else:
            time_elapsed = t - ts
            return self.risk_manager.get_discount_f_neg(time_elapsed)

    def get_discount_f_pos(self, t: MyDate, ts: MyDate):
        """
        Given:
         1) the current time "t"
         2) "ts" - the most recent time at which a negative-resulting test was made
                after an illness period, or alternatively - the most recent time of
                true recovery after illness.
        return the f_pos(time_interval) function,
        which is one of the terms of the discount factor

        :param t  current time step
        :param ts: most recent time of negative test after illness or the most recent time of true recovery
        :return: float in [0,1], the risk discount factor that should be applied to the weighted risks.
        """
        if ts is None:
            return 1.0
        else:
            time_elapsed = t - ts
            return self.risk_manager.get_discount_f_pos(time_elapsed)

    def init_nodes_attributes(self, organization_df: pandas.DataFrame, ext_risk_df: pandas.DataFrame, delta_df: pandas.DataFrame, group_risk_df: pandas.DataFrame, int_risk_df: pandas.DataFrame):
        """
        Sets each person a dictionary with the following values:
            'risk_ext': initial risk - is set to the provided external risk (as provided in the ext_risk_df) and normalized
            'ts_neg': time of the most recent test date of this person, which turned out to be a negative result (will be used for speculative discount factor)
            'ts_pos': time of the most recent test date of this person, which turned out to be a positive result (will be used for speculative discount factor)
            'tr': time of the most recent recovery (or None, if the person has never been ill or is still ill for his first time and it is in progress)
            'ext_infection_prob': current weight - the discounted risk
            'discount_factor': 1.0, - the factor that will multiply the 'risk_ext' to achieve 'ext_infection_prob'
            'speculative_discount_factor': 1.0, - a speculative approximation of 'discount_factor' - for external usage of agents
            'int_vect': internal risk vector,
            'delta': indidual delta factor,
            'state': the current health condition of the person,
            'days': the number of days in that health condition
            'time param': The duration of each infection state

        Sets each group a dictionary with the following 3 values:
            'w': current weight - equal to the sum of all the weights of the people associated with this group
            'vect': risk associated with the group and work environment
        This function updates both the self.nodes_attributes and the self.G

        :param ext_risk_df: a pandas dataframe carrying 3 columns of people id data,
                        followed by 1 column of 'Date of last COVID19 test' or 'תאריך בדיקה אחרון'
                        and followed the risk factor columns (as many as needed)
        :return:
        """
        columns = ext_risk_df.columns.tolist()
        covid_test_col_str = columns[3]
        covid_recovery_col_str = columns[4]
        # for External risk
        num_risk_factors = len(ext_risk_df.columns) - \
            self.num_ext_risk_df_columns_that_arent_risk_factors
        risk_factor_coefficients = self.risk_manager.get_coefficients('ext_risk',
                                                                      num_risk_factors)

        for personid in self.id_lst:
            ext_risk_vector = np.array(ext_risk_df.iloc[personid, self.num_ext_risk_df_columns_that_arent_risk_factors:],
                                       dtype=np.float32)
            ext_weighted_risk_vector = np.multiply(
                ext_risk_vector, risk_factor_coefficients)
            ext_static_risk = np.sum(ext_weighted_risk_vector)/5
            inter_risk_vector = np.array(int_risk_df.iloc[personid, self.num_int_risk_df_columns_that_arent_risk_factors:],
                                         dtype=np.float32)
            # TODO: use dedicated coefficients for internal risk factors. Enhance the risk manager if necessary
            int_risk_factor_coefficients = self.risk_manager.get_coefficients('int_risk',
                                                                              inter_risk_vector.size)
            pers_int_risk = np.sum(np.multiply(
                inter_risk_vector, int_risk_factor_coefficients))/5
            personal_test_date_str = ext_risk_df[covid_test_col_str][personid]
            personal_test_date = MyDate(
                strdate=personal_test_date_str) if personal_test_date_str not in ['None', "N\\A"] else None
            latest_recovery_date_str = ext_risk_df[covid_recovery_col_str][personid]
            if "\\" in latest_recovery_date_str:
                latest_recovery_date_str = latest_recovery_date_str.replace(
                    "\\", "-")
            latest_recovery_date = MyDate(
                strdate=latest_recovery_date_str) if latest_recovery_date_str not in ["None", "N\\A"] else None
            personal_delta_factor = delta_df['Spread'][personid]
            current_health_condition = delta_df['Status'][personid].upper()
            state_matrice = np.array(
                [0, 1], np.float32) if current_health_condition == "SUSCEPTIBLE" else np.array([1, 0], np.float32)
            age = organization_df['Age'][personid]
            symptom_prob = 0.4  # max(0.1, min(0.7, age/100))

            self.nodes_attributes[personid] = {'speculative_discount_factor': 1.0,
                                               'discount_factor': 1.0,
                                               'risk_ext': ext_static_risk,
                                               'risk_int': pers_int_risk,
                                               'delta': personal_delta_factor,
                                               'state': current_health_condition,
                                               'state_matrice': state_matrice,
                                               'days': 0,
                                               'isolated': [False, 0],
                                               'symptom_prob': symptom_prob,
                                               'int_infection_prob': 0.0,
                                               'ext_infection_prob': 0.0,
                                               'ts_neg': None,
                                               'ts_pos': None,
                                               'tr': latest_recovery_date,
                                               'time_params': self.get_illness_time_realization(),
                                               'D_iso': set(),
                                               'D_cont': set()
                                               }  # combine state and days_count
        for groupid, group in enumerate(self.group_lst):
            grp_risk_vector = np.array(group_risk_df.iloc[groupid, self.num_group_risk_df_columns_that_arent_risk_factors:],
                                       dtype=np.float32)
            # TODO: use dedicated coefficients for groups' risk factors. Enhance the risk manager if necessary
            grp_risk_factor_coefficients = self.risk_manager.get_coefficients('group_risk',
                                                                              grp_risk_vector.size)
            grp_static_risk = np.sum(np.multiply(
                grp_risk_vector, grp_risk_factor_coefficients)) / 5
            self.nodes_attributes[group] = {
                'w': 0.0, 'grp_risk': grp_static_risk}

        # this recalculates the group weights and updates the graph
        self.update_weights(current_date=self.current_date)

    def get_illness_time_realization(self):
        """
        draws the T1,T2,T3,T4,T5 random variables at random from the pre-defined distributions
        """
        t1 = ceil(abs(np.random.normal(
            self.risk_manager.t1, self.risk_manager.s1)))
        t2 = ceil(abs(np.random.normal(
            self.risk_manager.t2, self.risk_manager.s2)))
        t3 = ceil(abs(np.random.normal(
            self.risk_manager.t3, self.risk_manager.s3)))
        t4 = self.risk_manager.t4 + \
            ceil(abs(np.random.exponential(1 / self.risk_manager.s4)))
        t5 = ceil(abs(np.random.normal(
            self.risk_manager.t5, self.risk_manager.s5)))
        return t1, t2, t3, t4, t5

    def update_test_date(self, list_of_personid_testresult_pairs: List[Tuple[int, str]], test_date: MyDate):
        """
        Update the state of the sampled_person_lst such that their sampling time is
        update to the provided "sampling_time_step" value.

        This function updates both the self.nodes_attributes and the self.G

        :param list_of_personid_testresult_pairs:list of 2-tuples
        :param test_date: a date at which the person was tested
        :return:
        """

        for personid, test_result in list_of_personid_testresult_pairs:
            if test_result == 'P':
                self.nodes_attributes[personid]['ts_pos'] = test_date
            elif test_result == 'N':
                self.nodes_attributes[personid]['ts_neg'] = test_date
            else:
                raise ValueError("test result of person {} was caught being {}, whereas only P or N "
                                 "are plausible values. Any explanations?".format(personid, test_result))
        nx.set_node_attributes(self.G, self.nodes_attributes)

    def update_weights(self, current_date: MyDate):
        """
        (1) Update the weights (by discounting the risks) of all the people in the organization
        (2) Update the discount_factors (f_neg, f_pos) of all the people in the organization, based on
                   the last disease testing date and the latest recovery date relative to the "current_date".
        (3) Recalculates the weights of all the groups, as a consequence of the peoples' weight update.

        The peoples' weight update is according to the following strategy:
        if the person was not sampled or it was sampled more than 10 days ago --> his weight will be his initial_risk
        if the person was sampled during the last week --> his weight is 0.0
        if the person was sampled

        This function updates both the self.nodes_attributes and the self.G

        :param current_date: a date for which the weights of the people (and of the groups)
                             should be recalculated. The recalculation will be a result of
                             a new time intervals between the date each person was tested,
                             and the current_date (due to a discount factor).
        """

        # (1+2) recalculate the person discount factors and weights
        for personid in self.id_lst:

            # speculative discount factor for the usage of the external agents - TODO move it outside in the future. This belongs to the Agent class
            speculative_f_neg = self.get_discount_f_neg(
                current_date, self.nodes_attributes[personid]['ts_neg'])
            speculative_f_pos = self.get_discount_f_pos(
                current_date, self.nodes_attributes[personid]['ts_pos'])
            speculative_discount_factor = speculative_f_pos * speculative_f_neg \
                * (1 - int(self.nodes_attributes[personid]['isolated'][0]))
            self.nodes_attributes[personid]['speculative_discount_factor'] = speculative_discount_factor

            # discount factor of the model
            f_pos = self.get_discount_f_pos(
                current_date, self.nodes_attributes[personid]['tr'])
            discount_factor = f_pos * \
                (1-int(self.nodes_attributes[personid]['isolated'][0]))
            self.nodes_attributes[personid]['discount_factor'] = discount_factor

            if not self.nodes_attributes[personid]['isolated'][0]:
                self.nodes_attributes[personid]['ext_infection_prob'] = discount_factor * \
                    self.nodes_attributes[personid]['risk_ext']

        # (3) recalculate the group weights
        for group in self.group_lst:
            ext_probs = (self.nodes_attributes[personid]['ext_infection_prob']
                         for personid in self.G.neighbors(group))
            int_probs = (self.nodes_attributes[personid]['int_infection_prob']
                         for personid in self.G.neighbors(group))
            self.nodes_attributes[group]['w'] = sum(ext_probs) + sum(int_probs)

            # May be greater that 1?

        # (finally) plug all the node_attributes back into the graph structure.
        nx.set_node_attributes(self.G, self.nodes_attributes)

    def get_groups_of_people(self, personid_lst, format="dict"):
        """
        Given a list of people, determine the groups that are associated with them.
        :param person_lst: list of strings, with a person (node) name in each item.
        :param format: "list" - return only the list of the group names, that are covered by the people
                       "dict" - return a dictionary of the groups that this people is associated with,
                                 The returned dictionary has group_names as keys and people count as value
        :return: dictionary of group:num_sampled_person
        """
        sampled_groups_dict = {}
        for group_lst_of_person in [list(self.G.neighbors(personid)) for personid in personid_lst]:
            for group in group_lst_of_person:
                if group in sampled_groups_dict:
                    sampled_groups_dict[group] += 1
                else:
                    sampled_groups_dict[group] = 1

        if format == "dict":
            return sampled_groups_dict
        elif format == "list":
            return list(sampled_groups_dict.keys())
        else:
            raise TypeError("For the format keyword 'format' the only acceptable "
                            "values are 'dict' or 'list'. Given:{}".format(format))

    def get_people_of_one_group(self, group):
        assert group in self.group_lst
        return list(self.G.neighbors(group))

    def draw(self, node_size=200, marked_node_list=None, output_dir=None, output_filename=None, output_type="png",
             keep_fig_open=False, figsize=(16, 24), margins=0.2, font_size=12, crop_center=False):
        """
        Draws (using matplotlib) the bipartite graph representing the assignment of sets of people to the sets of groups.
        :param font_size:
        :param margins:
        :param figsize:
        :param node_size: size of a vertex in the plot that will be produced
        :param marked_nodes: a subset of the person list and group_lists. These vertices will be marked with red
        :param output_dir - if none, then no file will be produced, otherwise, an image file will be produced in this directory
        :param output_filename - if none, then no file will be produced, otherwise, an image file will be produced carrying this name
        :param crop_center: True if you want to crop the huge white border around the figure.
        :param keep_fig_open - if True then the figure will remainopen after the function returns
        :return:
        """
        if marked_node_list is None:
            marked_node_list = []
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig = plt.figure(figsize=figsize)
        if type(margins) == tuple:
            plt.margins(y=margins[0], x=margins[1])
        else:
            plt.margins(margins)
        ax = plt.gca() if not (output_dir is None or output_filename is None) else None
        node_positions = nx.bipartite_layout(
            self.G, nodes=self.id_lst, scale=1, )
        attrributes_positions = {k: (v[0], -0.008 + v[1])
                                 for k, v in node_positions.items()}

        nx.draw_networkx_nodes(self.G, node_positions,
                               nodelist=self.group_lst,
                               node_color='g',
                               node_size=node_size,
                               font_size=font_size,
                               alpha=0.8,
                               ax=ax)
        nx.draw_networkx_nodes(self.G, node_positions,
                               nodelist=self.id_lst,
                               node_color='b',
                               node_size=node_size,
                               font_size=font_size,
                               alpha=0.8,
                               ax=ax)
        i = 0
        color_palette = ['gray', 'green', 'orange', 'red']
        for marked_nodes in marked_node_list:
            selected_positions_to_mark = {}
            # list of edges of a format (person,group)
            selected_edge_to_mark = []
            for node_name in marked_nodes:
                selected_positions_to_mark[node_name] = node_positions[node_name]
                if node_name in self.id_lst:
                    for group_name in self.G.neighbors(node_name):
                        selected_edge_to_mark.append((node_name, group_name))

            nx.draw_networkx_nodes(self.G, node_positions,
                                   nodelist=selected_positions_to_mark,
                                   node_color=color_palette[i],
                                   node_size=node_size * 2,
                                   font_size=font_size,
                                   alpha=0.8,
                                   ax=ax)
            nx.draw_networkx_edges(self.G, node_positions,
                                   edgelist=selected_edge_to_mark,
                                   width=2.0, edge_color=color_palette[i],
                                   font_size=font_size,
                                   style='solid',
                                   ax=ax)
            i += 1
        lines = [Line2D([0, 1], [0, 1], color=clr)
                 for clr in color_palette]
        labels = ['Infected', 'detectable', 'contagious', 'symptomatic']
        plt.legend(lines, labels, loc='lower left')
        nx.draw(self.G, node_positions,
                with_labels=True, node_color="skyblue",
                font_size=font_size,
                alpha=0.5, linewidths=2, ax=ax)

        node_to_attr_string_dict = {}
        w_attributes = nx.get_node_attributes(self.G, 'ext_infection_prob')
        w_attributes.update(nx.get_node_attributes(self.G, 'w'))
        r_attributes = nx.get_node_attributes(self.G, 'risk_ext')
        # ts_attributes = nx.get_node_attributes(self.G, 'ts_neg')
        for personid in self.id_lst:
            node_to_attr_string_dict[personid] = self.person_idx_to_name_dict[personid]
            node_to_attr_string_dict[personid] += ", (w={:.2f} ".format(
                w_attributes[personid])
            node_to_attr_string_dict[personid] += ", r={:.2f})".format(
                r_attributes[personid])
            # ts_str = str(
            #     ts_attributes[personid]) if ts_attributes[personid] != -1 else "None"
            # node_to_attr_string_dict[personid
            #                          ] += ", ts={})".format(ts_str)
        for group in self.group_lst:
            node_to_attr_string_dict[group] = "(w={:.2f}) ".format(
                w_attributes[group])

        nx.draw_networkx_labels(self.G, attrributes_positions,
                                labels=node_to_attr_string_dict, font_size=font_size,)
        if output_dir is not None and output_filename is not None:
            # if "selected" in output_filename:
            #     fig_title = output_filename[6:].replace(":", " = ").replace("_", " people ")
            # elif "before" in output_filename:
            #     fig_title = output_filename[6:].replace(":"," = ").replace("_"," ") + " selection"
            try:
                picid = int(output_filename[-4:])
                if picid % 2 == 1:
                    fig_title = "t = {}, before test candidate selection".format(
                        (picid-1)//2)
                else:
                    fig_title = "t = {}, after test candidate selection".format(
                        (picid - 2) // 2)
                ax.set_title(fig_title, fontsize=40)
            except ValueError:
                pass
            if crop_center:
                ax.set_axis_off()
                fig.subplots_adjust(left=0, bottom=0, right=0,
                                    top=0, wspace=0, hspace=0)
            # ax.legend(loc='lower left',)
            fig.savefig(os.path.join(
                output_dir, output_filename+"."+output_type))
        if not keep_fig_open:
            plt.close(fig)

    def print_coverage_per_group(self, sampled_personid_lst: List[int], normalize_coverage: bool = True):
        """
        Given a list of sampled people, this function prints to the std-output the coverage c(e) obtained in each
        group e (individually).
        :param normalize_coverage: bool - set to True if a coverage of each group should be normalized by
                                          by the weight of the group.
        :param sampled_personid_lst: list of person's id
        """
        print("Coverage per group:")
        for group in self.group_lst:
            group_member_id_lst = self.get_people_of_one_group(group)
            group_coverage = 0.0
            group_num_sampled_person = 0
            for personid in group_member_id_lst:
                if personid in sampled_personid_lst:
                    group_coverage += self.nodes_attributes[personid]['ext_infection_prob']
                    group_num_sampled_person += 1
            if normalize_coverage:
                group_cov_norm = self.nodes_attributes[group]['w'] if self.nodes_attributes[group]['w'] != 0 else 1
                group_coverage = group_coverage / group_cov_norm
            print(' coverage({}) = {:.3f} ({}/{} person)'.format(group, group_coverage, group_num_sampled_person,
                                                                 len(group_member_id_lst)))

    def common_groups(self, personid_x: int, personid_y: int) -> List:
        """
        Given two people, determine a list of groups which both people are associated with.
        :param personid_x:- int, ID of person x
        :param person_y:- int, ID of person y
        """
        all_group_related = self.get_groups_of_people([personid_x, personid_y])
        return [group for group in list(all_group_related.keys()) if all_group_related[group] == 2]

    def probability_x_inf_y_per_group(self, group: str, personid_x: int, personid_y: int, pd_dict: dict) -> float:
        """
        **Ideally, you should only check probability_x_inf_y_per_group if person_x is positive/infected
        Both person_x and person_y belong to this same group/department.
        The probability pd_x_inf_y is a function of the int_risk_vector and the group_risk_vector and the coefficient_vector provided by the RiskManager

        :param group: str - common group to person_x and person_y
        :param personid_x: int - ID of the person infected 
        :param personid_y: int - ID of the concerned
        :param pd_dict: dict - dictionary of Pds

        Considering that a person_x who is not in isolation can transmit the disease on between the period [contagious,not_contagious],
        the following strategy will be considered:
        (1) Before the start of 'contagious' state and (3) after the end of 'not contagious' state, namely contagion period:
         The probability is very small, assumed 0. So is the probability, when person_x is isolated.
        (2) Between the two previously mentioned states, the probability with be calculated
        as defined and multiplied by the delta (may be recalculated)

        """
        if not self.nodes_attributes[personid_x]['isolated'][0] \
                and not self.nodes_attributes[personid_y]['isolated'][0] \
                and (self.nodes_attributes[personid_x]['state'] == "CONTAGIOUS"
                     or self.nodes_attributes[personid_x]['state'] == 'SYMPTOMATIC'):
            if pd_dict is None:
                pd_dict = {}
            keys = (personid_x, personid_y, group)
            if keys not in pd_dict:
                delta_x = self.nodes_attributes[personid_x]['delta']
                lambda_y = self.nodes_attributes[personid_y]['discount_factor']
                risk_int_y = self.nodes_attributes[personid_y]['risk_int']
                grp_risk = self.nodes_attributes[group]['grp_risk']
                pd_dict[keys] = min(
                    1.0, delta_x * lambda_y * grp_risk * risk_int_y)
            return pd_dict[keys]
        return 0.0

    def cond_probability_x_inf_y(self, personid_x: int, personid_y: int, pd_dict: dict):
        assert personid_x != personid_y
        assert personid_x in self.id_lst, "Invalid name input: {}".format(
            personid_x)
        assert personid_y in self.id_lst, "Invalid name input: {}".format(
            personid_y)
        big_pi = 1  # To compute the complement of the probability
        for grp in self.common_groups(personid_x, personid_y):
            big_pi *= 1 - \
                self.probability_x_inf_y_per_group(
                    grp, personid_x, personid_y, pd_dict)
        return 1 - big_pi  # 0.0 will be returned if there are no common groups for x and y

    def probability_all_y(self, personid_x: str, pd_dict: dict):
        """ Given a person x and a date, this function prints the updated probability that every person_y that works in at least one department/group
        with person_x got infected by this latter, given person_x is infected
        :param person_x: str - name of the person infected
        :param pd_dict: dict - dictionary of Pds
        """
        assert personid_x in self.id_lst, "{} is not in the registrered IDs".format(
            personid_x)
        x_groups = self.get_groups_of_people([personid_x], format="list")
        prob_dict = {}
       # key = (personid_x, personid_y, group) , value = probability
        for group in x_groups:
            list_of_people_id = self.get_people_of_one_group(
                group)  # without the person_x
            list_of_people_id.remove(personid_x)
            if list_of_people_id == None:
                continue
            for personid in list_of_people_id:
                key = self.person_idx_to_name_dict[personid]
                if not key in prob_dict:
                    prob_dict[key] = self.cond_probability_x_inf_y(
                        personid_x, personid, pd_dict)
                continue
        print(" Here is the probability of infection of every person related to {} by at least one departement".format(personid_x))
        for person in list(prob_dict.keys()):
            print("{} : {}".format(person, prob_dict[person]))

    def probability_y_pos(self, personid_y: int, pd_dict: dict) -> float:
        """ Given a name of an employer and a memoisation dictionary pd_dict, the function return the probability the  person is positive
        """
        assert personid_y in self.id_lst, "{} is not in the registrered IDs".format(
            personid_y)
        prob_y_infected_externally = self.nodes_attributes[personid_y]['ext_infection_prob']
        y_groups = self.get_groups_of_people([personid_y], format="list")
        keep_track = []
        complement_prob_y_inf_internally = 1
        for group in y_groups:
            list_of_peopleid = self.get_people_of_one_group(group)
            # without the personid_y themselves
            list_of_peopleid.remove(personid_y)
            if list_of_peopleid == None:
                continue
            for personid in list_of_peopleid:
                if not (personid in keep_track):
                    keep_track.append(personid)
                    complement_prob_y_inf_internally *= 1 - self.cond_probability_x_inf_y(
                        personid, personid_y, pd_dict)
                continue
        self.nodes_attributes[personid_y]['int_infection_prob'] = 1 - \
            complement_prob_y_inf_internally
        return 1 - (1 - prob_y_infected_externally) * complement_prob_y_inf_internally

    def prob_group1_inf_group2(self, group1: str, group2: str, pd_dict: dict):
        """Given the health condition of a group1's members, on current date, return the probability group2 counts at at least one infected
            only because of people infected in group1
            :param group1: - str: the name of the group with infected people
            :param group2: - str: the targeted group (ideally with no infection, o/w only on non-infected people)
            :param pd_dict: dict - dictionary of Pds
        """
        if group1 == "ALL":
            peopleId_in_group2 = [pid for pid in self.get_people_of_one_group(
                group2) if self.nodes_attributes[pid]['state'] == "SUSCEPTIBLE"]

            if not peopleId_in_group2:
                return 0
            comp_group_inf_given_infected_group = 1
            for personid in peopleId_in_group2:
                comp_group_inf_given_infected_group *= 1 - \
                    self.probability_y_pos(personid, pd_dict)
            return 1 - comp_group_inf_given_infected_group
        else:
            peopleId_in_group2 = [pid for pid in self.get_people_of_one_group(
                group2) if self.nodes_attributes[pid]['state'] == "SUSCEPTIBLE"]
            contagious_in_group1 = [pid for pid in self.get_people_of_one_group(
                group1) if self.nodes_attributes[pid]['state'] in ['CONTAGIOUS', 'SYMPTOMATIC']]
            if peopleId_in_group2 == [] or contagious_in_group1 == []:  # an empty set, no need to compute
                return 0
            comp_group_inf_given_infected_group = 1
            for personid in peopleId_in_group2:
                comp_pers_inf_given_infected_group = 1
                for contagiousId in contagious_in_group1:
                    comp_pers_inf_given_infected_group *= 1-self.cond_probability_x_inf_y(
                        contagiousId, personid, pd_dict)
                comp_group_inf_given_infected_group *= comp_pers_inf_given_infected_group
            return 1 - comp_group_inf_given_infected_group
# Recall:
# overview_list = []  #  [[INFECTED,DETECTABLE,CONTAGIOUS,SYMPTOMATIC,RECOVERED]]  index = duration
# groups_view_dict= {} #keys = group's name, value= [[inf,contagious]] with index = duration

    def record_current_state(self, overview_list: list, groups_view_dict: dict, duration: int, pd_dict: dict):
        symptomatic_list = []
        contagious_list = []
        detectable_list = []
        infected_list = []
        overview_list.append([0, 0, 0, 0, 0])
        for group in self.group_lst:
            if group not in groups_view_dict:
                groups_view_dict[group] = []
            if duration == 0:
                groups_view_dict.get(group).append([0, 0])
            else:
                groups_view_dict.get(group).append(
                    [self.prob_group1_inf_group2('ALL', group, pd_dict), 0])
        for personid in self.id_lst:
            if self.nodes_attributes[personid]['state'] != "SUSCEPTIBLE":
                if self.nodes_attributes[personid]['state'] == 'INFECTED':
                    overview_list[duration][0] += 1
                    infected_list.append(personid)
                elif self.nodes_attributes[personid]['state'] == 'DETECTABLE':
                    overview_list[duration][1] += 1
                    detectable_list.append(personid)
                elif self.nodes_attributes[personid]['state'] == 'CONTAGIOUS':
                    overview_list[duration][2] += 1
                    contagious_list.append(personid)
                    temp = self.nodes_attributes[personid]['D_cont']
                    temp.add(duration)
                    person_groups = self.get_groups_of_people(
                        [personid], format="list")
                    for group in person_groups:
                        temp = groups_view_dict[group][duration]
                        temp[1] += 1
                elif self.nodes_attributes[personid]['state'] == 'SYMPTOMATIC':
                    overview_list[duration][3] += 1
                    temp = self.nodes_attributes[personid]['D_cont']
                    temp.add(duration)
                    symptomatic_list.append(personid)
                elif self.nodes_attributes[personid]['state'] == "RECOVERED":
                    overview_list[duration][4] += 1
        return infected_list, detectable_list, symptomatic_list, contagious_list

    def transition_function(self, first_case: bool, pd_dict: dict) -> bool:
        """
        Given parameters, the function allows transition to the next state of each person/worker
            :param time_variables: -tuple, of time variables
            :param pd_dict: -dict, memoisation of Pds
            :param first_case: - bool, True if there is already a first in the institution
        """
        rng = np.random  # .default_rng(seed)  Kostya changed to np.random since the initialization of the random generator should be done elsewhere, and not before every transition.
        self.update_weights(self.current_date)
        infected_daily_count = 0
        day_total_ill_count = 0
        for personid in self.id_lst:

            if self.nodes_attributes[personid]['state'] not in ['SUSCEPTIBLE', 'RECOVERED']:
                t1, t2, t3, t4, t5 = self.nodes_attributes[personid]['time_params']
                day_total_ill_count += 1
                if self.nodes_attributes[personid]['state'] == 'INFECTED':
                    self.nodes_attributes[personid]['days'] += 1
                    if self.nodes_attributes[personid]['days'] == t1:
                        self.nodes_attributes[personid]['state'] = 'DETECTABLE'
                        self.nodes_attributes[personid]['days'] = 0
                elif self.nodes_attributes[personid]['state'] == 'DETECTABLE':
                    self.nodes_attributes[personid]['days'] += 1
                    if self.nodes_attributes[personid]['days'] == t2:
                        self.nodes_attributes[personid]['state'] = 'CONTAGIOUS'
                        self.nodes_attributes[personid]['days'] = 0
                elif self.nodes_attributes[personid]['state'] == 'CONTAGIOUS':
                    self.nodes_attributes[personid]['days'] += 1
                    if self.nodes_attributes[personid]['days'] == t4 and t4 != t3:
                        if rng.binomial(1, p=self.nodes_attributes[personid]['symptom_prob']) == 1:
                            # the person shows symptoms
                            self.nodes_attributes[personid]['state'] = 'SYMPTOMATIC'
                            self.nodes_attributes[personid]['days'] = 0
                    else:  # considering also t4 != t3
                        if self.nodes_attributes[personid]['days'] == t3:
                            self.nodes_attributes[personid]['state'] = 'RECOVERED'
                            self.nodes_attributes[personid]['days'] = 0
                            self.nodes_attributes[personid]['state_matrice'] = np.array(
                                [0, 1], np.float32)
                            self.nodes_attributes[personid]['int_infection_prob'] = 0.0
                            self.nodes_attributes[personid]['tr'] = self.current_date
                elif self.nodes_attributes[personid]['state'] == 'SYMPTOMATIC':
                    self.nodes_attributes[personid]['days'] += 1
                    if self.nodes_attributes[personid]['days'] == t5:
                        self.nodes_attributes[personid]['state'] = 'RECOVERED'
                        self.nodes_attributes[personid]['days'] = 0
                        self.nodes_attributes[personid]['state_matrice'] = np.array(
                            [0, 1], np.float32)
                        self.nodes_attributes[personid]['int_infection_prob'] = 0.0
                        self.nodes_attributes[personid]['tr'] = self.current_date
            else:
                if not self.nodes_attributes[personid]['isolated'][0]:
                    # init the state with the current one
                    # current_state = self.nodes_attributes[personid]['state_matrice']
                    # if not first_case:
                    #     prob_of_infection = self.nodes_attributes[personid]['ext_infection_prob']
                    # else:
                    #     prob_of_infection = self.probability_y_pos(personid, pd_dict)
                    # transition_matrix = np.array([[1, 0], [prob_of_infection, 1-prob_of_infection]], np.float32)
                    # new_state = current_state.dot(transition_matrix)
                    ext_infected = rng.binomial(
                        n=1, p=self.nodes_attributes[personid]['ext_infection_prob'])
                    int_infected = rng.binomial(
                        n=1, p=self.nodes_attributes[personid]['int_infection_prob'])
                    if int_infected == 1 or ext_infected == 1:
                        # This person is beginning his illnes right now!
                        # We draw the t1,...,t5 now (at the beginning of the illness session)
                        self.nodes_attributes[personid]['time_params'] = self.get_illness_time_realization(
                        )
                        self.nodes_attributes[personid]['infection_source'] = self.infection_source(
                            [ext_infected, int_infected])
                        self.nodes_attributes[personid]['state'] = 'INFECTED'
                        self.nodes_attributes[personid]['state_matrice'] = np.array(
                            [1, 0], np.float32)
                        infected_daily_count += 1
                        # since you are already infected
                        self.nodes_attributes[personid]['int_infection_prob'] = 0.0
                        self.nodes_attributes[personid]['days'] = 0
                        if not first_case:
                            first_case = True
                    # else:
                    #     self.nodes_attributes[personid]['state_matrice'] = new_state
            if self.nodes_attributes[personid]['isolated'][0]:
                temp = self.nodes_attributes[personid]['isolated']
                temp[1] += 1
        if infected_daily_count > self.day_peak_max[0]:
            self.day_peak_max = (infected_daily_count,
                                 MyDate(self.current_date.strdate))
        if day_total_ill_count > self.day_total_ill_peak_max[0]:
            self.day_total_ill_peak_max = (
                day_total_ill_count, MyDate(self.current_date.strdate))
        self.current_date.increment()
        return first_case

    def speculative_person_weight(self, personid, agent_type: str, test_overview):
        supported_agent_types = ["RFG", "Optimization"]
        if agent_type in supported_agent_types:
            lambda_x = self.nodes_attributes[personid]['speculative_discount_factor']
            inter_risk = self.nodes_attributes[personid]['risk_int']
            ext_risk = self.nodes_attributes[personid]['risk_ext']
            grp_contrb = 0
            pers_group = self.get_groups_of_people([personid])
            for group in pers_group:
                grp_contrb += (1/len(pers_group)) \
                    * (self.group_assess(group, test_overview)+1)\
                    * self.nodes_attributes[group]['grp_risk']
            return lambda_x * (ext_risk + inter_risk * grp_contrb)
        else:
            raise NotImplementedError("Speculative person weight is implemented for the desired agent_type {}. "
                                      "Only types {} are supported.".format(agent_type, supported_agent_types))

    def group_assess(self, group, test_overview):
        return np.array(test_overview[group]).sum()

    def test_result(self, state: str, days: int) -> str:
        """
        :param state: str- the current state of the person sampled
        :param days: int- how lon the person have been in that state, useful only if the person is infected
        Return "P" for "Positive" or "N" for "Negative" if the person is in an infected state using the
        probability of false positive/false negative and the how long the person is the infected state
        """
        rng = np.random
        positive = 0
        # How will days affected that probability?
        if state == 'INFECTED':
            positive = rng.choice([0, 1], 1, p=[
                                  self.risk_manager.p_test_error_infected, 1-self.risk_manager.p_test_error_infected])[0]
        elif state == 'DETECTABLE':
            positive = rng.choice([0, 1], 1, p=[
                                  self.risk_manager.p_test_error_detectable, 1-self.risk_manager.p_test_error_detectable])[0]
        elif state == 'CONTAGIOUS':
            positive = rng.choice([0, 1], 1, p=[
                                  self.risk_manager.p_test_error_contagious, 1-self.risk_manager.p_test_error_contagious])[0]
        elif state == 'SYMPTOMATIC':
            positive = rng.choice([0, 1], 1, p=[
                                  self.risk_manager.p_test_error_symptomatic, 1-self.risk_manager.p_test_error_symptomatic])[0]
        elif state == "RECOVERED":
            positive = rng.choice([0, 1], 1, p=[
                                  1-self.risk_manager.p_test_error_non_contagious, self.risk_manager.p_test_error_non_contagious])[0]
        else:
            positive = rng.choice([0, 1], 1, p=[
                                  1-self.risk_manager.p_test_error_idle, self.risk_manager.p_test_error_idle])[0]
        return 'P' if positive == 1 else 'N'

    def infection_source(self, inf_list: list):
        """
        :param: inf_list - list, contained three element: infected,ext_infected,int_infected which are either 0 or 1
          - infected being 1 means the person is infected o/w 0
          - int_infected being 1 means the person's infection (if infected) is a result of internal factors o/w 0
          - ext_infected being 1 means the person's infection (if infected) is a result of external factors o/w 0
        """
        if inf_list[0] != inf_list[1]:
            if inf_list[0] == 1:
                self.infection_stat[1] = self.infection_stat[1]+1
                return "Ext_infection"
            self.infection_stat[2] = self.infection_stat[2]+1
            return "Int_infection"
        self.infection_stat[0] = self.infection_stat[0]+1
        return "INFECTED"
