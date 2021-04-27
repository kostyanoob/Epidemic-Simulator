from simulator import Epidemic_simulator  # class
from agent import agent_factory
import numpy
import random
import itertools
import argparse
import os
import json
from Util.plot import safely_dump_dictionary
from tqdm import tqdm
from explore_data import get_metrics_graphs
import pdb


def clear_redundand_parameter_combinations(args, parameter_combinations):
    parameter_combinations_ret = []
    for (seed, agent_type, budget, test_range, simulation_duration, outputs, rfi_raw_name) in parameter_combinations:

        # cancel Symp and nopolicy agents' budget variation and test_range variation
        if agent_type in ["nopolicy", "Symp"] and (budget != args.budgets[0]
                                                   or test_range != args.test_ranges[0]
                                                   or rfi_raw_name != args.return_from_isolations[0]):
            continue

        if agent_type != "RFG" and test_range != args.test_ranges[0]:
            continue

        # yield seed, agent_type, budget, test_range, simulation_duration, outputs, rfi_raw_name
        parameter_combinations_ret.append((seed, agent_type, budget, test_range, simulation_duration, outputs, rfi_raw_name))
    return parameter_combinations_ret


def main(args):
    batch_summary = {}
    seeds = range(args.num_repetitions)
    # have the same outputs for every agent
    outputs_list = [args.output_products]
    parameter_combinations = itertools.product(seeds, args.agent_types, args.budgets, args.test_ranges,
                                               args.simulation_durations, outputs_list, args.return_from_isolations)
    parameter_combinations = clear_redundand_parameter_combinations(args, parameter_combinations)
    pbar = tqdm(total=len(parameter_combinations), desc="Completed runs")
    type_bugdet = []
    agents_sum = []
    for (seed, agent_type, budget, test_range, simulation_duration, outputs, rfi_raw_name) in parameter_combinations:

        # Randomness initialization
        numpy.random.seed(seed)
        random.seed(seed)
        add_AT_B = "{}_{}".format(agent_type, budget)
        if add_AT_B not in type_bugdet:
            type_bugdet.append(add_AT_B)
        if agent_type not in agents_sum:
            agents_sum.append(agent_type)
        # Set up paths for output products of the current run
        run_dirname = '{}__AT_{}__RFI_{}__B_{}__SD_{}__S_{}'.format(args.run_dir_name_prefix, agent_type, rfi_raw_name, budget,
                                                            simulation_duration, seed)
        if 'args' in args.output_products:
            safely_dump_dictionary(os.path.join(
                args.output_dir, run_dirname), 'args.txt', args.__dict__)

        agent = agent_factory(agent_type, rfi_raw_name)
        my_simulator = Epidemic_simulator(
            args.simulation_inputs_filepath, args.output_dir, run_dirname)
        run_summary = my_simulator.run_simulation(agent=agent,
                                                  budget=budget,
                                                  test_range=test_range,
                                                  simulation_duration=simulation_duration,
                                                  outputs=outputs,
                                                  verbose=False)
        batch_summary[run_dirname] = run_summary
        pbar.update(1)
    batch_summary['type_bugdet'] = type_bugdet
    batch_summary['agents_sum'] = list(agents_sum)
    pbar.close()
    safely_dump_dictionary(args.output_dir, args.run_dir_name_prefix +
                           '_batch_summary.txt', batch_summary, verbose=True)
    get_metrics_graphs(batch_summary, args.output_dir, agent_order=args.agent_types)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="COVID-19 Simulator - Batched version. Runs several parametric "
                                                 "simulations and writes one summary dictionary with its keys being "
                                                 "the unique run names (which combine the various parameters and "
                                                 "random seeds used for this run) and the values being the run_summary "
                                                 "dictionary. That is, the result is a dictionary of dictionaries.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-agent_types', '-AT', type=str, nargs='*',
                        default=['Optimization', 'RFG', 'Rand', 'Symp', 'nopolicy'],
                        help='Agent type list, space separated. Can have values in "Optimization", "RFG", "Rand", '
                             '"Symp", "nopolicy"')
    parser.add_argument('-return_from_isolations', '-RFI', type=str, nargs='*',
                        default=["Delay_4-Interval_2"],
                        help='Determines the policy of repeat testing of the isolated people. For example,'
                             'choosing Delay_9-Interval_7, sets the behavior to wait 9 days and then perform a test '
                             'every 7 days until a negative result (which will release the person from isolation)')
    parser.add_argument('-simulation_durations', '-SD', type=int, nargs='*',
                        default=[300],
                        help='Simulation durations list, space separated. Natural number.')
    parser.add_argument('-test_ranges', '-TR', type=int, nargs='*',
                        default=[5],
                        help='Test range in days for obtaining person weights for RFG and Optimization agents. '
                             'Space separated natural numbers.')
    parser.add_argument('-budgets', '-B', type=int, nargs='*',
                        default=[1, 5, 10, 15, 20, 25, 30, 40, 50],
                        help='List of daily test budget allocated for the agent. space separated natural numbers.')
    parser.add_argument('-output_dir', '-OD', type=str,
                        default='Results',
                        help='Directory to contain the batch summary file as well as the run directories of the '
                             'individual runs (if requested at least one output product.)')
    parser.add_argument('-run_dir_name_prefix', '-RDNP', type=str,
                        default='Yeshiva',
                        help="Prefix of the batch_summary.txt file that will conclude the run. It will also be a "
                             "prefix of the individual directories to contain each individual run's products (in "
                             "case at least one output product was requested).")
    parser.add_argument('-simulation_inputs_filepath', '-SIF', type=str,
                        default='simulation_inputs_two_clusters.txt',
                        help="Path to a file containing some extra simulation parameters")
    parser.add_argument('-output_products', '-OP', action='store', dest='output_products',
                        type=str, nargs='*',
                        default=[],
                        help="List of space separated names of plots/text-files that you would like to be produced in "
                             "each individual run. No need to request here the batch_summary.txt, since it will be  "
                             "generated anyways. The full list of the plots currently supported are: args, "
                             "run_summary Daily_logs, Isolation_logs, states_graphs, all_states_breakdown_daily, "
                             "illness_states_only_breakdown_daily, infection_probability_per_group_daily, "
                             "ill_person_count_per_group_daily.")
    parser.add_argument('-num_repetitions', '-NR', type=int,
                        default=1,
                        help='Number of repetitions to perform the simulation for each set of parameters. A random '
                             'seed number between 0 and num_repetitions-1 (inclusive) will be assigned for each run.')

    args = parser.parse_args()
    main(args)
