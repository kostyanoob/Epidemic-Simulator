python main.py \
 -agent_type nopolicy \
 -return_from_isolation Delay_4-Interval_2 \
 -budget 10 \
 -output_dir Results \
 -run_dir_name_prefix Quickstart \
 -simulation_inputs_filepath \
 simulation_inputs_two_clusters.txt \
 -simulation_duration 100 \
 -seed 0 \
 -output_products args run_summary Daily_logs Isolation_logs all_states_breakdown_daily illness_states_only_breakdown_daily infection_probability_per_group_daily ill_person_count_per_group_daily
 
python main.py \
 -agent_type Symp \
 -return_from_isolation Delay_4-Interval_2 \
 -budget 10 \
 -output_dir Results \
 -run_dir_name_prefix Quickstart \
 -simulation_inputs_filepath \
 simulation_inputs_two_clusters.txt \
 -simulation_duration 100 \
 -seed 0 \
 -output_products args run_summary Daily_logs Isolation_logs all_states_breakdown_daily illness_states_only_breakdown_daily infection_probability_per_group_daily ill_person_count_per_group_daily
 
python main.py \
 -agent_type Rand \
 -return_from_isolation Delay_4-Interval_2 \
 -budget 10 \
 -output_dir Results \
 -run_dir_name_prefix Quickstart \
 -simulation_inputs_filepath \
 simulation_inputs_two_clusters.txt \
 -simulation_duration 100 \
 -seed 0 \
 -output_products args run_summary Daily_logs Isolation_logs all_states_breakdown_daily illness_states_only_breakdown_daily infection_probability_per_group_daily ill_person_count_per_group_daily
 
python main.py \
 -agent_type RFG \
 -return_from_isolation Delay_4-Interval_2 \
 -budget 10 \
 -output_dir Results \
 -run_dir_name_prefix Quickstart \
 -simulation_inputs_filepath \
 simulation_inputs_two_clusters.txt \
 -simulation_duration 100 \
 -seed 0 \
 -output_products args run_summary Daily_logs Isolation_logs all_states_breakdown_daily illness_states_only_breakdown_daily infection_probability_per_group_daily ill_person_count_per_group_daily
 
python main.py \
 -agent_type Optimization \
 -return_from_isolation Delay_4-Interval_2 \
 -budget 10 \
 -output_dir Results \
 -run_dir_name_prefix Quickstart \
 -simulation_inputs_filepath \
 simulation_inputs_two_clusters.txt \
 -simulation_duration 100 \
 -seed 0 \
 -output_products args run_summary Daily_logs Isolation_logs all_states_breakdown_daily illness_states_only_breakdown_daily infection_probability_per_group_daily ill_person_count_per_group_daily
 
 