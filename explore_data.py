import json
from typing import List

import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from Util.plot import graph_feature, safely_dump_dictionary
import argparse

metric_list_glob = ['Peak_Morbidity', 'Total_Morbidity', '-Infected_externally',
                    '-Infected_internally', 'Isolated_people', "mPQE", "GQE", "ill_isolated", "healthy_isolated",
                    "ill_free"]


def extract_dict_from_file(filepath) -> dict:
    with open(filepath) as f:
        content = json.load(f)
        return content


def recover_batch_summary_from_run_summaries(output_dir: str, summary_filename: str, metric_list: List[str]):
    """
    :param output_dir: str, a path to the directory containing partial runs.
    :param summary_filename: str, a filename to be created inside output_dir
    :param metric_list: List of parameters that must reside in each run_summary_dict.txt (as dictionary keys) in order to
                        refer to this run_summary_dict as complete and to take it into an account when constructing
                        batch_summary
    """

    batch_summary = {}
    agenttype_bugdet_set = set()
    agents_sum_set = set()
    num_aqcuired = 0
    num_skipped = 0
    run_prefix = None

    # For every run-directory inside output_dir:
    for run_dir_entry in os.scandir(output_dir):
        if os.path.isdir(run_dir_entry):
            if "__" not in run_dir_entry.name:
                num_skipped += 1
                print('Skipping dir {} - name not formatted as a proper run-name (should be of a format '
                      '"prefix__parameters...")'.format(run_dir_entry.path))
                continue

            run_name_parsed_list = run_dir_entry.name.split(
                "__")  # {}__AT_{}__RFI_{}__B_{}__SD_{}__S_{}
            run_prefix_current = run_name_parsed_list[0]

            if run_prefix is None:
                run_prefix = run_prefix_current
            elif run_prefix != run_prefix_current:
                raise ValueError("In the path {}, encountered differently prefixed runs ({} vs {}). Unable to "
                                 "continue. If you choose to recover a batch_summary.txt, you must make sure "
                                 "that the directory {} will contain run-directories prefixed exactly the same."
                                 "".format(output_dir, run_prefix, run_prefix_current, output_dir))

            # Check that the run directory contains a valid run_summary.txt with a proper dictionary in it
            path_to_summary_txt = os.path.join(
                run_dir_entry.path, "run_summary.txt")
            if os.path.exists(path_to_summary_txt):
                run_summary_dict = extract_dict_from_file(path_to_summary_txt)
                if all(key in run_summary_dict for key in metric_list):
                    num_aqcuired += 1
                    batch_summary[run_dir_entry.name] = run_summary_dict
                    run_params = {kv.split("_")[0]: kv.split(
                        "_")[1] for kv in run_name_parsed_list[1:]}
                    agents_sum_set.add(run_params["AT"])
                    agenttype_bugdet_set.add("{}_{}".format(
                        run_params["AT"], run_params["B"]))
                else:
                    num_skipped += 1
                    print("Skipping file {} - incomplete file (not all the parameters are specified in "
                          "it)".format(path_to_summary_txt))
            else:
                num_skipped += 1
                print(
                    "Skipping dir {} - doesn't contain run_summary.txt)".format(run_dir_entry.path))

    batch_summary['type_bugdet'] = list(agenttype_bugdet_set)
    batch_summary['agents_sum'] = list(agents_sum_set)

    print("Acquired {} run_summary files (skipped {}).".format(
        num_aqcuired, num_skipped))
    safely_dump_dictionary(output_dir, summary_filename,
                           batch_summary, verbose=True)

    #  check it for containing a legal run_summary_dict.txt
    #  read the run_summary_dict and store it
    #  batch_summary[sub-dirname] = run_summary_dict
    #  count it if everything was OK

    # store the batch_summary to file
    # print to the screen how many seeds were collected
    # perform the countings that Koffi performed at the end of the batch_summary dictionary construction.


def get_metrics_graphs(summary, output_dir, metric_list=None, features=None, additional_metrics=None, agent_order=None):
    """
    :param :str or dict- the batch summary file path(str) or summary dictionary
    :param output_dir:str - the output dir. path
    :param metric_list:list - list of parameters to extract from the batch summary dictionary
    :param features :list - list of metrics that will be plot individually
    :param additional_metrics:list - list of additional metrics (will be plot as stacked bar plots)
    :param agent_order: list of agent-type names according to the order you wish them to be ordered in the plots.

    """
    if len(output_dir) > 0:
        if not os.path.exists(os.path.join(output_dir, "Batch_Metrics")):
            os.makedirs(os.path.join(output_dir, "Batch_Metrics"))

    plot_filename_extensions = ["png", "eps"]

    if agent_order is None:
        agent_order = ['Optimization', 'RFG', 'Rand', 'Symp', 'nopolicy']
    agent_order = {agent_type: i for i, agent_type in enumerate(agent_order)}

    if metric_list is None:
        metric_list = metric_list_glob

    if isinstance(summary, str):
        summary = extract_dict_from_file(summary)

    if features is None:
        features = ['Peak_Morbidity', 'Total_Morbidity', '-Infected_externally',
                    '-Infected_internally', 'Isolated_people', "mPQE", "GQE"]

    if additional_metrics is None:
        additional_metrics = ["ill_isolated", "healthy_isolated", "ill_free"]

    agent_types = sorted(summary['agents_sum'], key=lambda at: agent_order[at])
    type_budgets = [elemt.split("_") for elemt in summary['type_bugdet']]

    for key in summary:
        if key not in ['type_bugdet', 'agents_sum']:
            summary[key].pop("PQE_x")
            up_dict = {}
            params = key.split("__")
            # skip = False
            for i in range(1, len(params)):
                # if skip: #
                #     skip = False
                #     continue
                param = params[i].split("_")
                up_dict[param[0]] = param[1]
                # if param[0] == "RFI":
                #     up_dict[param[0]] += "_"+param[2]+"__"+params[i+1]
                #     skip = True
            summary[key].update(up_dict)

    rev_summary = {}
    for param in ['AT', 'B'] + metric_list:
        values = []
        for elt in summary:
            if elt not in ['type_bugdet', 'agents_sum']:
                values.append(summary[elt][param])
        rev_summary[param] = values

    sum_data = pd.DataFrame(rev_summary, index=list(
        range(len(list(rev_summary.values())[0]))))

    mean_summary = []
    std_summary = []

    for agent_type, budget in type_budgets:
        at_budget = sum_data[sum_data['AT'].eq(
            agent_type) & sum_data['B'].eq(budget)]
        save = at_budget[at_budget.columns[:2]].iloc[0]
        std_at_budget = at_budget.std(axis=0).to_frame().fillna(0).T
        mean_at_budget = at_budget.mean(axis=0).to_frame().T

        # correct the df
        mean_at_budget['AT'] = save['AT']
        mean_at_budget['B'] = save['B']
        std_at_budget['AT'] = save['AT']
        std_at_budget['B'] = save['B']

        # append for concat
        mean_summary.append(mean_at_budget)
        std_summary.append(std_at_budget)

    mean_summary = pd.concat(mean_summary, axis=0, ignore_index=True)
    std_summary = pd.concat(std_summary, axis=0, ignore_index=True)

    # x_data = []  # number of existing budgets budgets
    unique_budgets = mean_summary['B'].astype(int).unique()
    for feature in features:
        ys = []
        xs = []
        stds = []
        for agent_type in agent_types:
            at_group = mean_summary[mean_summary['AT'].eq(agent_type)]
            at_group_std = std_summary[std_summary['AT'].eq(agent_type)]
            # if len(x_data) == 0:
            #     x_data = at_group["B"].tolist()
            budgets = at_group["B"].astype(int).tolist()
            values = at_group[feature].astype(float).tolist()
            values_std = at_group_std[feature].astype(float).tolist()

            if agent_type in ["nopolicy", "Symp"]:
                budgets = unique_budgets
                values = values * len(unique_budgets)
                values_std = [0] * len(unique_budgets)

            ys.append([y for _, y in sorted(zip(budgets, values))])
            stds.append([s for _, s in sorted(zip(budgets, values_std))])
            xs.append(list(str(xint) for xint in sorted(budgets)))

        #  label hack, TODO solve it later by actual renaming of the metrics in the run_summary and batch_summary files
        if feature in ['Peak_Morbidity', 'Total_Morbidity']:
            ylabel = feature.replace('_', ' ') + ' (num. of people)'
        else:
            ylabel = feature.replace('_', ' ')

        graph_feature(xs, ys, stds, agent_types, x_label="Bugdet", y_label=ylabel,
                      output_dir=os.path.join(output_dir, "Batch_Metrics"), filename=feature,
                      file_extensions=plot_filename_extensions, font_size=16)

    mean_summary = mean_summary.drop(features, axis=1)

    fig, axes = plt.subplots(nrows=1, ncols=len(
        agent_types), figsize=(25, 10), sharey=True)

    num = 0
    for agent_type in agent_types:
        font_size = 24
        show_legend = agent_type == "nopolicy"
        target_axes = axes[num] if len(agent_types) > 1 else axes
        new_df = mean_summary[mean_summary["AT"]
                              == agent_type].drop(['AT'], axis=1)
        new_df['B'] = new_df['B'].astype(int)
        new_df.sort_values(by='B', inplace=True)
        plt.rcParams.update({'font.size': font_size})
        new_df.plot(ax=target_axes, x="B", kind='bar', width=0.38, stacked=True, color=[
                    'green', 'gray', 'red'], legend=show_legend)
        target_axes.set_xlabel("Budget", fontsize=font_size)
        target_axes.set_ylabel(
            "Percentage of all people-days", fontsize=font_size)
        # if agent_type == "nopolicy":
        #     target_axes.legend(title=agent_type, title_fontsize='medium', fancybox=True)
        # else:
        target_axes.set_title(agent_type)
        num += 1

    for filename_extension in plot_filename_extensions:
        plt.savefig(os.path.join(output_dir, "Batch_Metrics",
                                 "Health_ill_isolated_free_barplot."+filename_extension),
                    bbox_inches="tight", pad_inches=0)
    print("All the plots were written to " + output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Exploring batch-run results and visualizing them",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-output_dir', '-OD', type=str,
                        default="Results",
                        help='Directory where all the run results supposed to exist, and where the "Batch_Metrics" '
                             'directory will be created by this script, and will contain all the summary plots of all '
                             'the runs in the batch')
    parser.add_argument('-summary_filename', '-SF', type=str,
                        default="batch_summary.txt",
                        help='A filename of a batch_summary file (contatining a dictionary of all run summaries, '
                             'that is supposed to be inside the output_dir')
    parser.add_argument('--recover', '--R', action="store_true",
                        help='If used, then the script will evaluate the "output_dir" directory contents, and '
                             'reconstruct the batch_summary text file, and then it will use the existing seeds there '
                             'to create the summary plots (and store them in a Batch_Metrics directory). Note: to '
                             'actually recover anything, the individual simulator runs should have been executed '
                             'with option -OP run_summary, otherwise there will be nothing to recover the information '
                             'from)')

    args = parser.parse_args()

    if args.recover:
        recover_batch_summary_from_run_summaries(
            args.output_dir, args.summary_filename, metric_list_glob)

    get_metrics_graphs(os.path.join(
        args.output_dir, args.summary_filename), args.output_dir, metric_list_glob)
    # get_metrics_graphs("ResultsY", "ResultsY/Yeshiv_B123_batch_summary.txt")
