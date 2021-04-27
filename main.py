from simulator import Epidemic_simulator  # class
from agent import agent_factory
import numpy, random
import argparse, os, json
from Util.plot import safely_dump_dictionary

parser = argparse.ArgumentParser(description="COVID-19 Simulator",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-agent_type', '-AT', choices=["Optimization", "RFG", "Rand", "Symp", "nopolicy"],
                    default='nopolicy',
                    help='agent type')
parser.add_argument('-return_from_isolations', '-RFI', type=str,
                    default="Delay_4-Interval_2",
                    help='Determines the policy of repeat testing of the isolated people. For example,'
                         'choosing Delay_9-Interval_7, sets the behavior to wait 9 days and the perform a test '
                         'every 7 days until a negative result (which will release the person from isolation).')
parser.add_argument('-simulation_duration', '-SD', type=int,
                    default=100,
                    help='in days')
parser.add_argument('-test_range', '-TR', type=int,
                    default=5,
                    help='Test range in days for obtaining person weights for RFG and Optimization agents. In the '
                         'article this value is denoted by T.')
parser.add_argument('-budget', '-B', type=int,
                    default=20,
                    help='daily test budget allocated for the agent')
parser.add_argument('-output_dir', '-OD', type=str,
                    default='Results',
                    help='Directory containing the results')
parser.add_argument('-run_dir_name_prefix', '-RDNP', type=str,
                    default='Yeshiva',
                    help="Prefix of the individual directory to contain this run's products. This directory will be "
                         "created inside the output_dir")
parser.add_argument('-simulation_inputs_filepath', '-SIF', type=str,
                    default='simulation_inputs_two_clusters.txt',
                    help="Path to a file containing some extra simulation parameters")
parser.add_argument('-seed', '-S', type=int,
                    default=0,
                    help='Random seed, to initialize the generators in the numpy and the random packages')
parser.add_argument('-output_products', '-OP', type=str, nargs='*',
                    default=["args", "run_summary", "Daily_logs", "Isolation_logs", "all_states_breakdown_daily",
                             "illness_states_only_breakdown_daily", "infection_probability_per_group_daily",
                             "ill_person_count_per_group_daily"],
                    help="List of space separated names of plots/text-files that you would like to be produced in the "
                         "run. The full list of the plots currentlyst supported are: args, run_summary, Daily_logs,"
                         "Isolation_logs, states_graphs, all_states_breakdown_daily, "
                         "illness_states_only_breakdown_daily, infection_probability_per_group_daily, "
                         "ill_person_count_per_group_daily.")

args = parser.parse_args()

# Randomness initialization
numpy.random.seed(args.seed)
random.seed(args.seed)

# Set up paths for output products
run_dirname = '{}__AT_{}__RFI_{}__B_{}__SD_{}__S_{}'.format(args.run_dir_name_prefix, args.agent_type,
                                                            args.return_from_isolations, args.budget,
                                                            args.simulation_duration, args.seed)
if 'args' in args.output_products:
    safely_dump_dictionary(os.path.join(args.output_dir, run_dirname), 'args.txt', args.__dict__)

# Go
# batch_summary = {}
agent = agent_factory(args.agent_type, args.return_from_isolations)
my_simulator = Epidemic_simulator(args.simulation_inputs_filepath, args.output_dir, run_dirname)
run_summary = my_simulator.run_simulation(agent=agent,
                                          budget=args.budget,
                                          test_range=args.test_range,
                                          simulation_duration=args.simulation_duration,
                                          outputs=args.output_products,
                                          verbose=True)
# batch_summary[run_dirname] = run_summary
