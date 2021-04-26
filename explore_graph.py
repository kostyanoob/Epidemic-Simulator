"""
This script reads the 4 datasets and computes graph-related characteristics of
the underlying community graphs of these dataasets. Among the characteristics,
there are the graph diameter, the expansion ratio of random groups of different
sizes.
(expansion ratio of a group is by how much a the size given group smaller than the
size of the group of all their neighbors)
"""


from simulator import Epidemic_simulator
import networkx as nx
import numpy as np
np.random.seed(0)
output_dir = "Dataset"
run_dirname = "Graph_Metrics"
input_filepath_list = ["simulation_inputs_singleschool.txt", "simulation_inputs_singleschool_wandering.txt",
                       "simulation_inputs_multischool_families.txt", "simulation_inputs_multischool_friends.txt"]
num_people_list = [150, 150, 1000, 1000]
for input_filepath, num_people in zip(input_filepath_list, num_people_list):
    dataset_name = input_filepath.split(".")[0][18:]
    G = Epidemic_simulator(input_filepath, output_dir, run_dirname).institution.G
    A = nx.adjacency_matrix(G)  # A is sparse !

    # Construct the inter-people graph, rather than bipartite graph that involves no connection between people
    sqA = A*A
    Asq_people = sqA[:num_people, :num_people]
    Gpeople = nx.from_scipy_sparse_matrix(Asq_people)

    # Diameter
    diameter = nx.algorithms.distance_measures.diameter(Gpeople)

    # Closeness
    closeness_vec_per_node = np.array([v for _,v in nx.algorithms.centrality.closeness_centrality(Gpeople).items()])
    all_pairs_average_min_distance = (1 / closeness_vec_per_node).mean()
    node_expansion_of_the_cleaner = nx.algorithms.cuts.node_expansion(Gpeople, [2])
    node_expansion_of_the_first_student = nx.algorithms.cuts.node_expansion(Gpeople, [0])
    node_expansion_of_the_cleaner_and_first_student = nx.algorithms.cuts.node_expansion(Gpeople, [0,2])
    node_expansion_of_two_students_from_different_classes = nx.algorithms.cuts.node_expansion(Gpeople, [1,3])

    # Expansions
    expansions_ngroups = 100
    expansion_group_sizes = [5, 10, 20, 50]
    expansions = {}
    for group_size in expansion_group_sizes:
        groups = np.random.randint(0, high=num_people, size=[expansions_ngroups, group_size], dtype='l')
        expansion_per_group = (nx.algorithms.cuts.node_expansion(Gpeople, groups[i,:]) for i in range(expansions_ngroups))
        expansions[group_size] = sum(expansion_per_group) / float(expansions_ngroups)

    print("Dataset {}".format(dataset_name))
    print("   diameter: {}".format(diameter))
    print("   all pairs average min distance: {}".format(all_pairs_average_min_distance))
    print("   node_expansion_of_the_cleaner: {}".format(node_expansion_of_the_cleaner))
    print("   node_expansion_of_the_first_student: {}".format(node_expansion_of_the_first_student))
    print("   node_expansion_of_the_cleaner_and_first_student: {}".format(node_expansion_of_the_cleaner_and_first_student))
    print("   node_expansion_of_two_students_from_different_classes: {}".format(node_expansion_of_two_students_from_different_classes))
    for group_size, expansion in expansions.items():
        print("   Average expansion of {} groups of size {} is: {}".format(expansions_ngroups, group_size, expansion))