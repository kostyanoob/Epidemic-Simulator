from Util.spreadsheet import edit_speadsheet_xlsx as edit_sp, read_main_spreadsheet as read_sp
from Util.plot import safely_dump_dictionary
from RiskManager import RiskManager  # class
from Institution import Institution  # class
from LinearProgramming import SelectCandidatesForTest
from MyDate import MyDate  # class
from agent import Agent, Risk_factor_greedy_agent, No_policy_agent, Symptom_based_agent
import pandas as pd
import numpy as np
from typing import Tuple, List, Union
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm
from sys import platform
import io
import os
import json
import pdb


class Epidemic_simulator:
    def __init__(self, parameters_file_path: str, output_dir: str, runname_dir: str):
        # Flat Data
        self.institution = None
        self.output_path = os.path.join(output_dir, runname_dir)
        self.SId = runname_dir
        self.logs_file_path = ''
        self.isolated_people = 0
        self.isolation_periods_count = 0
        self.isolation_periods_in_progress = 0
        self.isolation_periods_completed = 0
        self.d_iso = []
        self.d_cont = []
        self.pqe_x = []
        self.gqe = 1
        if parameters_file_path != "":
            self.load(parameters_file_path)

    def load(self, parameters_file_path: str):
        file = None
        try:
            file = open(parameters_file_path, 'r')
            parameters_list = file.readlines()
            parameters = [param.rstrip() for param in parameters_list]
            assert len(
                parameters) == 3, "There number of parameters in simulation_inputs.txt is different than 3. Exiting..."
            organization_df, ext_risk_df, group_risk_df, int_risk_df, delta_df, msg = read_sp(
                parameters[0])
            assert not organization_df.empty, "error occurred while trying to load the simulator"
            self.logs_file_path = parameters[0]
            risk_manager = RiskManager(parameters[1])
            start = MyDate(parameters[2])
            self.institution = Institution(
                organization_df, ext_risk_df, group_risk_df, int_risk_df, delta_df, start, risk_manager)
        finally:
            if file is not None:
                file.close()

    def run_simulation(self, agent: Agent, budget: int, test_range: int, simulation_duration: int, outputs: list,
                       verbose=True):
        """
        Given a list of parameters, a simulation is run and
        the outputs/graphs are store with names provided in the outputs list.
        return:
        """

        if len(outputs) > 0:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        stop_date = MyDate(self.institution.current_date.strdate)
        stop_date.increment(add=simulation_duration)
        # [[INFECTED,DETECTABLE,CONTAGIOUS,SYMPTOMATIC,RECOVERED]]  index = duration
        overview_list = []
        # keys = group's name, value= [[inf,contagious]] with index = duration
        groups_view_dict = {}
        pd_dict = {}
        if verbose:
            print("COVID Simulation Model")
        run_summary = self.simulate(agent, budget, test_range, stop_date, overview_list, groups_view_dict, pd_dict,
                                    outputs)
        if verbose:
            print("Done")
        return run_summary

    def get_graphs(self, overview_list: list, groups_view_dict: dict, total: int, outputs: list):
        """Given the collected data on the evolution of the Infection in the institution, return a graph and
        store it in the  the graph with given output name
        :param overview_list: - nested list; raw data from the simulation about the general evolution
        :param groups_view_dict: -dict group's data on specific features
        :param total: -int the number of people in the whole intitution
        :param outputs: - list of the desired graphs to be produced. The keys relevant to this function are:
                                     "all_states_breakdown_daily",
                                     "illness_states_only_breakdown_daily",
                                     "infection_probability_per_group_daily",
                                     "ill_person_count_per_group_daily"
        :return:
        """

        overview = []
        # five main states (INFECTED,DETECTABLE,CONTAGIOUS,SYMPTOMATIC,RECOVERED)
        for j in range(5):
            overview.append([0]*len(overview_list))
            for i in range(len(overview_list)):  # loop on dates/durations
                overview[j][i] = overview_list[i][j]

        # graph the overview of the institution
        if "all_states_breakdown_daily" in outputs:
            self.graph_overview(overview, os.path.join(self.output_path, "all_states_breakdown_daily"),
                                plot_title="All states breakdown - daily",
                                filename_extensions=["png", "eps"])

        if "illness_states_only_breakdown_daily" in outputs:
            self.graph_overview(overview[:-1], os.path.join(self.output_path, "illness_states_only_breakdown_daily"),
                                plot_title="Illness states only breakdown - daily",
                                filename_extensions=["png", "eps"])

        # prep other features' data
        group_infection = {}
        group_contagion = {}
        for group, data in list(groups_view_dict.items()):
            contagion = []
            infection = []
            for t in range(len(data)):
                infection.append(data[t][0])
                contagion.append(data[t][1])
            group_infection[group] = infection
            group_contagion[group] = contagion

        # Daily Group infection probabilities Graph  and Daily Group contagion evolution Graph
        if "infection_probability_per_group_daily" in outputs:
            self.graph_feature(group_infection, "Daily Infection probability of each group",
                               "Simulation time step (days)", "Probabilities",
                               os.path.join(
                                   self.output_path, "infection_probability_per_group_daily"),
                               max_y=None, font_size=15)

        if "ill_person_count_per_group_daily" in outputs:
            self.graph_feature(group_contagion, "Daily ill people count in each group\n(what is the definition of ill?)",
                               "Simulation time step (days)", "People count",
                               os.path.join(self.output_path,
                                            "ill_person_count_per_group_daily"),
                               max_y=None, font_size=15)

    def graph_overview(self, overview: list, output_path: str, plot_title: str, filename_extensions=None):
        """
        Given the collected data on the evolution of the Infection in the institution, return a graph and
        store it in the  the graph with given output name
        :param overview: - nested list
        :output_path:-str the name of the output file/figure with the extention
        :return:
        """
        if filename_extensions is None:
            filename_extensions = ["png"]
        # Overview Graphing
        x = range(len(overview[0]))
        length = len(overview)
        reverse = [[]]*length
        for i in range(length):
            reverse[i] = overview[length-1-i]
        labels = ['RECOVERED', 'SYMPTOMATIC',
                  'CONTAGIOUS', 'DETECTABLE', 'INFECTED']
        plt.figure()
        plt.stackplot(
            x, reverse, labels=labels[-length:], colors=['purple', 'red', 'orange', 'green', 'gray'])
        plt.legend(loc='best')
        #plt.title(plot_title, loc='center',fontsize=12, color='Orange', fontweight=0)
        plt.xlabel("Simulation time step (days)")
        plt.ylabel("People count")
        for fext in filename_extensions:
            plt.savefig(output_path+"."+fext)
        plt.close()

    def graph_feature(self, feature: dict, suptitle: str, x_label: str, y_label: str, output_path: str,
                      font_size=None, max_y=None):
        """
        Given the collected data on the evolution of the *feature in the institution, return a graph and
        store it in the  the graph with given output name
        :param feature: - dict
        :param suptitle: - str
        :param x_label:- str x-axis label
        :param y_label:- str y-axis label
        :param output_path: - str file name with the extention
        :param legend_loc: -str position of the legend on the graph
        :return:
        """
        if font_size is None:
            font_size = 11
        plt.rcParams['font.size'] = font_size

        plt.figure()
        ys = list(feature.values())
        plt.style.use('seaborn-white')
        palette = sns.color_palette("Set1")
        line_style = ["--", ":", "-.", "-"]
        markers = ["o", "^", "<", ">", "x", "X", "s", "d", "D",
                   "v", "p", "P", ".", ",", "1", "2", "3", "4", "8"]
        num = 0
        for y in ys:
            title = list(feature.keys())[num]
            x = range(len(y))
            plt.plot(x, y, marker=markers[num % len(markers)],
                     linestyle=line_style[num % len(line_style)],
                     color=palette[num % len(palette)],
                     linewidth=1, alpha=0.9, label=title)
            # Add legend
            plt.legend(loc='best', prop={'size': font_size-1})
            num += 1

        if max_y is not None:
            plt.ylim(0, max_y)

        plt.xlabel(x_label, fontsize=font_size-1)
        plt.ylabel(y_label, fontsize=font_size)
        #plt.suptitle(suptitle, fontsize=12,fontweight=0, color='orange')
        plt.savefig(output_path)
        plt.close()

    def simulate(self, agent: Agent, bugdet: int, test_range: int, stop_date: MyDate, overview_list: list,
                 groups_view_dict: dict, pd_dict: dict, outputs: list):
        """ Given the institution on current date, simulate and save to archive every day state until stop_date
        :param agent: -Agent,  the type of agent that is regulating the simulation
        :param stop_date: -MyDate,  the date to stop the simulation
        :param overview_list: -list, (Nested)list where the raw data on the whole institution after the simulation
        :param groups_view_dict: -dict, where the raw data on the each group concerning some *features after the simulation
        :param pd_dict: -dict, memoization of Pds
        :param outputs: - list, containing the names of the statistics products
        if the respective statistic output is not wanted, or a filename otherwise.
        """
        duration = 0
        first_case = False
        test_overview = {}
        start_date = MyDate(self.institution.current_date.strdate)
        for group in self.institution.group_lst:
            test_overview[group] = [0]*test_range
        with tqdm(total=stop_date-self.institution.current_date, ascii=True, desc="Simulated days") as pbar:
            while(self.institution.current_date - stop_date) < 0:
                daily_logs = []
                isol_logs = []
                for personid in self.institution.id_lst:
                    if self.institution.nodes_attributes[personid]['state'] in ["SUSCEPTIBLE", "RECOVERED"]:
                        srecord = (self.institution.nodes_attributes[personid]['state'],
                                   self.institution.nodes_attributes[personid]['state_matrice'][0],
                                   self.institution.nodes_attributes[personid]['int_infection_prob'],
                                   self.institution.nodes_attributes[personid]['ext_infection_prob'])
                    else:
                        if self.institution.nodes_attributes[personid]['state'] == "INFECTED":
                            srecord = self.institution.nodes_attributes[personid]['infection_source']
                        else:
                            srecord = self.institution.nodes_attributes[personid]['state']

                    daily_logs.append(srecord)
                day = 'Day_{}'.format(duration)
                self.institution.log[day] = daily_logs
                infected_list, detectable_list, symptomatic_list, contagious_list = self.institution.record_current_state(
                    overview_list, groups_view_dict, duration, pd_dict)

                if 'states_graphs' in outputs:
                    self.institution.draw(marked_node_list=[infected_list, detectable_list, symptomatic_list, contagious_list],
                                          output_dir=os.path.join(self.output_path, 'states_graphs'), output_filename=day)
                # ACTION REQUIRED FROM AGENT (Maybe not at day 0)
                if duration != 0:
                    # * "no request" "apply_test"
                    request = agent.request()
                    if request == 0 or request == "no request":
                        pass
                    elif request == 1 or request == "apply_test":
                        # get testing type and budget
                        test_request = agent.apply_test(bugdet)
                        sampled = self.sampling(
                            test_request, test_overview, test_range)
                        tested, tested_str = self.test_sample(sampled)
                    decision = agent.decision()
                    if decision == 100:
                        pass
                    elif decision == 101:
                        assessing = {}
                        isolate = []
                        un_isolate = []
                        isolate.extend([ids for ids in symptomatic_list if not self.institution.nodes_attributes[ids]
                                        ['isolated'][0]])
                        if not (isinstance(agent, No_policy_agent) or isinstance(agent, Symptom_based_agent)):
                            if test_request[0] in ["RFG", "Rand", "Optimization"]:
                                # and result[0] not in symptomatic_list
                                infected = [result[0]
                                            for result in tested if result[1] == 'P']
                                not_infected = [result[0]
                                                for result in tested if result[1] == 'N']
                                isolate.extend(infected)
                                un_isolate.extend(not_infected)
                                # only RFG relies on institution.score() function, hence we want to update the assessing
                                if test_request[0] == "RFG":
                                    assessing.update(
                                        self.institution.get_groups_of_people(infected, "dict"))
                                    for group in assessing:
                                        temp = test_overview[group]
                                        temp[duration %
                                             test_range] = assessing[group]
                                    assessing.update(self.institution.get_groups_of_people(
                                        symptomatic_list, "dict"))
                        # red_groups = [groups for groups in list(
                        #    assessing.keys()) if assessing[groups] >= 5]
                        self.isolate_person(isolate)
                        self.isolate_person(un_isolate, reverse=True)

                        # // act on people who where isolate. Reverse Isolation/free if tested negative
                        # re_test_sampling = [
                        #     ids for ids in self.institution.id_lst if self.institution.nodes_attributes[ids]['isolated'][1] > 2 and self.institution.nodes_attributes[ids]['isolated'][1] % 7 == 2]
                        candidates_for_resampling = filter(
                            lambda ids: self.institution.nodes_attributes[ids]['state'] != "SYMPTOMATIC", self.institution.id_lst)
                        re_test_sampling = agent.get_re_test_list(
                            (ids, self.institution.nodes_attributes[ids]['isolated'][1]) for ids in candidates_for_resampling)
                        if len(re_test_sampling) != 0:
                            tested, tested_str = self.test_sample(
                                re_test_sampling)
                            free = [result[0]
                                    for result in tested if result[1] == 'N']
                            self.isolate_person(free, True)
                    isol_logs.extend([self.institution.nodes_attributes[ids]
                                      ['isolated'][1] for ids in self.institution.id_lst])
                    day = 'Day_{}'.format(duration)
                    self.institution.isol_log[day] = isol_logs

                first_case = self.institution.transition_function(
                    first_case, pd_dict)
                duration += 1
                pbar.update(1)
        infected_list, detectable_list, symptomatic_list, contagious_list = self.institution.record_current_state(
            overview_list, groups_view_dict, duration, pd_dict)

        if "states_graphs" in outputs:
            self.institution.draw(marked_node_list=[infected_list, detectable_list, symptomatic_list, contagious_list],
                                  output_dir=os.path.join(
                                      self.output_path, 'states_graphs'),
                                  output_filename='Day_{}'.format(duration))
        daily_logs = []
        isol_logs = []
        for personid in self.institution.id_lst:
            isol_logs.append(
                self.institution.nodes_attributes[personid]['isolated'][1])
            if self.institution.nodes_attributes[personid]['state'] in ["SUSCEPTIBLE", "RECOVERED"]:
                srecord = (self.institution.nodes_attributes[personid]['state'],
                           self.institution.nodes_attributes[personid]['state_matrice'][0],
                           self.institution.nodes_attributes[personid]['int_infection_prob'],
                           self.institution.nodes_attributes[personid]['ext_infection_prob'])
            else:
                if self.institution.nodes_attributes[personid]['state'] == "DETECTABLE":
                    srecord = self.institution.nodes_attributes[personid]['infection_source']
                else:
                    srecord = self.institution.nodes_attributes[personid]['state']
            daily_logs.append(srecord)
        day = 'Day_{}'.format(duration)
        self.institution.log[day] = daily_logs
        self.institution.isol_log[day] = isol_logs

        total_freq = len(self.institution.person_lst)
        self.get_graphs(overview_list, groups_view_dict, total_freq, outputs)
        return self.dumping(outputs, start_date)

    def dumping(self, outputs, start_date):
        """
        :param: list of strings, that specifies the desired textual/csv outputs.
                relevant values are:
                 'run_summary'
                 'Daily_logs'
                 'Isolation_logs'
        :Return: a dict, summary of the simulation for analytics
        """
        df1 = self.institution.log
        df1 = df1.drop(df1.columns[:4], 1)
        df2 = self.institution.isol_log
        df2 = df2.drop(df2.columns[:3], 1)
        self.iou_metrics(df1, df2)
        other_metric = self.other_metric(
            df1, df2, ['h_fr', 'il_isol', 'h_isol', "il_fr"])
        run_summary = {'Peak_Morbidity': self.institution.day_total_ill_peak_max[0],
                       '-Day_of_Peak_Morbidity': self.institution.day_total_ill_peak_max[1]-start_date,
                       'Peak_number_of_new_daily_infected_people': self.institution.day_peak_max[0],
                       '-Day_number_of_peak_new_infected': self.institution.day_peak_max[1]-start_date,
                       'Total_Morbidity': sum(self.institution.infection_stat),
                       '-Infected_externally': self.institution.infection_stat[1],
                       '-Infected_internally': self.institution.infection_stat[2],
                       '-Infected_from_both_sources_(undefined)': self.institution.infection_stat[0],
                       'Isolated_people': int(self.isolated_people),
                       'Isolation_periods': self.isolation_periods_count,  # self.isolate_count,
                       '-In_Progress': self.isolation_periods_in_progress,  # self.current_isolated,
                       '-Completed': self.isolation_periods_completed,  # self.free_count,
                       "GQE": self.gqe,
                       "mPQE": sum(self.pqe_x) / len(self.pqe_x),
                       "PQE_x": self.pqe_x
                       }

        run_summary.update(other_metric)

        if "run_summary" in outputs:
            with open(os.path.join(self.output_path, 'run_summary.txt'), 'w') as f:
                json.dump(run_summary, f, indent=2)
        if "Daily_logs" in outputs:
            self.institution.log.to_csv(
                os.path.join(self.output_path, "Daily_logs.csv"))
        if "Isolation_logs" in outputs:
            self.institution.isol_log.to_csv(
                os.path.join(self.output_path, "Isolation_logs.csv"))
        return run_summary

    def iou_metrics(self, daily_df, isol_df):
        """
        :param: daily_df - a preprocessed dataframe of daily_logs dataframe containing only days' columns
        :param: isol_df - a preprocessed dataframe of isol_logs dataframe containing only days' columns
        :return:
        """
        temp = daily_df.eq("CONTAGIOUS")
        temp2 = daily_df.eq("SYMPTOMATIC")
        daily_df = temp | temp2
        isol_df = isol_df.ne(0)
        self.isolated_people = isol_df.sum(axis=1).ne(0).sum()
        self.pqe_x = []
        for pid in self.institution.id_lst:
            d_cont = daily_df.loc[pid]
            d_iso = isol_df.loc[pid]
            self.pqe_x.append(((d_cont & d_iso).sum(
            ) + np.finfo(float).eps) / ((d_cont | d_iso).sum()+np.finfo(float).eps))

        self.gqe = ((daily_df & isol_df).sum(axis=0).sum() + np.finfo(float).eps) / \
                   ((daily_df | isol_df).sum(axis=0).sum()+np.finfo(float).eps)

    def other_metric(self, daily_df, isol_df, metrics: list):
        """
        :param: daily_df - a preprocessed dataframe of daily_logs dataframe containing only days' columns
        :param: isol_df - a preprocessed dataframe of isol_logs dataframe containing only days' columns
        :param: metrics- list, list of some intersection to be evaluated
        :return: other_metric-dict
        """
        daily_df = daily_df.astype(str)
        other_metric = {}
        temp = daily_df.apply(
            lambda cols: cols.str.startswith("('SUS"), axis=0)
        temp2 = daily_df.apply(
            lambda cols: cols.str.startswith("('REC"), axis=0)
        df1 = temp | temp2  # healthy
        df2 = isol_df.eq(0)  # not isolated
        size = int(df1.size)
        if 'h_fr' in metrics:
            other_metric['healhy_free'] = 100 * \
                int((df1 & df2).sum(axis=0).sum())/size

        if 'il_isol' in metrics:
            other_metric['ill_isolated'] = 100 * \
                int((~df1 & ~df2).sum(axis=0).sum())/size

        if 'h_isol' in metrics:
            other_metric['healthy_isolated'] = 100 * \
                int((df1 & ~df2).sum(axis=0).sum())/size

        if "il_fr" in metrics:
            other_metric['ill_free'] = 100 * \
                int((~df1 & df2).sum(axis=0).sum())/size

        return other_metric

    def sampling(self, agent_req: list, test_overview, test_range, target="all", target_type=None, target_ratio=None):
        agent_type, budget = agent_req[0], agent_req[1]
        # .default_rng(self.seed)
        rng = np.random
        if target_type is None or target_type == "individual":
            people_score = {}
            if target_type is None:
                target_people = [ids for ids in self.institution.id_lst if not self.institution.nodes_attributes[ids]
                                 ['isolated'][0] and self.institution.nodes_attributes[ids]['state'] != 'SYMPTOMATIC']
            else:
                target_people = [ids for ids in target if not self.institution.nodes_attributes[ids]
                                 ['isolated'][0] and self.institution.nodes_attributes[ids]['state'] != 'SYMPTOMATIC']
            final_list = []
            if agent_type == "RFG":
                leftover = budget-len(target_people)  # check surplus
                if leftover >= 0:
                    final_list = target_people
                    if leftover != 0:
                        surplus_ids = {ids: self.institution.nodes_attributes[ids]['isolated'][1]
                                       for ids in self.institution.id_lst if self.institution.nodes_attributes[ids]['isolated'][0]}
                        interest_ids = list(surplus_ids.keys())
                        sorted_insterest_id = sorted(
                            interest_ids, key=surplus_ids.get, reverse=True)
                        final_list.extend(sorted_insterest_id[:leftover])
                else:
                    for personid in target_people:
                        people_score[personid] = self.institution.speculative_person_weight(
                            personid, agent_type, test_overview)
                    interest_ids = list(people_score.keys())
                    sorted_insterest_id = sorted(
                        interest_ids, key=people_score.get, reverse=True)
                    final_list = sorted_insterest_id[:budget]
            elif agent_type == 'Rand':
                leftover = budget-len(target_people)  # check surplus
                if leftover >= 0:
                    final_list = target_people
                    if leftover != 0:
                        surplus_ids = [
                            ids for ids in self.institution.id_lst if self.institution.nodes_attributes[ids]['isolated'][0]]
                        if len(surplus_ids) <= leftover:
                            final_list.extend(surplus_ids)
                        else:
                            interest_ids = list(rng.choice(
                                surplus_ids, leftover, False))
                            final_list.extend(interest_ids)
                else:
                    final_list = list(rng.choice(target_people, budget, False))
            elif agent_type == 'Optimization':
                problem = SelectCandidatesForTest(test_budget=budget,
                                                  institution=self.institution,
                                                  target_people_ids=target_people,
                                                  normalized_coverage=True,
                                                  test_overview=test_overview)
                if platform == "linux" or platform == "linux2":
                    solver_path = "glpsol"
                elif platform == "darwin":
                    raise NotImplementedError(
                        "Optimization based agent is not supported on OS-X")
                elif platform == "win32":
                    solver_path = "Solvers/glpk-4.65/w64/glpsol.exe"

                final_list = problem.solve(path=solver_path, verbosity=0)
            else:
                raise NotImplementedError(
                    "Agent type {} is not supported".format(agent_type))

        elif target_type == "groups":
            sum_ratio = np.array(target_ratio).sum()
            assert sum_ratio == budget, "The sum of your required proportion per group is different than the bugdet"
            try:
                final_list = []
                i = 0
                for group in target:
                    assert group in self.institution.group_lst
                    target_people = [ids for ids in self.institution.get_people_of_one_group(
                        group) if self.institution.nodes_attributes[ids]['isolated'][0] and ids not in final_list]
                    if agent_type == "RFG":
                        people_score = {}
                        for personid in target_people:
                            people_score[personid] = self.institution.speculative_person_weight(
                                personid, agent_type, test_overview)
                        interest_ids = people_score.keys()
                        sorted_insterest_id = sorted(
                            interest_ids, key=people_score.get, reverse=True)
                        new_list = sorted_insterest_id[:target_ratio[i]]
                        i += 1
                        final_list.extend(new_list)
                    if agent_type == "Rand":
                        if target_ratio[i] >= len(target_people):
                            final_list.extend(target_people)
                            i += 1
                            continue
                        final_list.extend(
                            list(rng.choice(target_people, target_ratio[i], False)))
                        i += 1

            except AssertionError:
                print("There is an incorrect group name")
        return final_list

    def test_sample(self, sample: list):
        """
        Given a list of people, return a test result of each person in the sample.
        :param sample: - list of people
        :param output_type: - either list or string
        return: 
        """
        sample_test_result = []
        printing = "Test results"
        for personid in sample:
            assert personid in self.institution.id_lst, "The input ID #{} is not registered".format(
                personid)
            test_res = (personid, self.institution.test_result(
                self.institution.nodes_attributes[personid]['state'], self.institution.nodes_attributes[personid]['days']))
            sample_test_result.append(test_res)
            printing += "\n{} : {}".format(
                self.institution.person_idx_to_name_dict[test_res[0]], test_res[1])
        self.institution.update_test_date(
            sample_test_result, self.institution.current_date)
        return sample_test_result, printing

    def test_group(self, group_sample: list):
        """Given a list of group, return a test result of each person in the groups sampled.
        :param sample: - list of group
        :param output_type: 
        return:
        """
        group_test_result = {}
        for group in group_sample:
            assert group in self.institution.group_lst, "The group with name {} is not part of the group list".format(
                group)
            people_in_group = self.institution.get_people_of_one_group(group)
            group_test_result[group] = self.test_sample(people_in_group)
        return group_test_result

    def isolate_person(self, sample: list, reverse=False):
        """
            reverse = True <---> the sample list are the people to be released from isolation
            reverse = False <---> the sample list are the people to be isolated
        """
        if not reverse:
            for personid in sample:
                assert personid in self.institution.id_lst, "The input ID #{} is not registred".format(
                    personid)
                isisolated_isolationdays_tuple = self.institution.nodes_attributes[
                    personid]['isolated']
                isisolated_isolationdays_tuple[1] = isisolated_isolationdays_tuple[1]+1
                if not isisolated_isolationdays_tuple[0]:
                    isisolated_isolationdays_tuple[0] = True
                    self.institution.nodes_attributes[personid]['int_infection_prob'] = 0.0
                    self.isolation_periods_count += 1
                    self.isolation_periods_in_progress += 1
        else:
            for personid in sample:
                assert personid in self.institution.id_lst, "The input ID #{} is not registred".format(
                    personid)
                isisolated_isolationdays_tuple = self.institution.nodes_attributes[
                    personid]['isolated']
                if isisolated_isolationdays_tuple[0]:
                    isisolated_isolationdays_tuple[0] = False
                    isisolated_isolationdays_tuple[1] = 0
                    self.isolation_periods_completed += 1
                    self.isolation_periods_in_progress -= 1

    def isolate_group(self, group_sample, reverse=False):
        """
            reverse = True <---> the  group_sample are the groups whose people to be released from isolation
            reverse = False <---> the  group_sample are the groups whose people to be isolated
        """
        for group in group_sample:
            assert group in self.institution.group_lst, "The group with name {} is not part of the group list".format(
                group)
            peopleId_in_group = self.institution.get_people_of_one_group(
                group)
            return self.isolate_person(peopleId_in_group, reverse)
