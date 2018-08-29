import multiprocessing
import logging
import os
import argparse
import pickle
import time
import gc

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import graph_tool.all as gt

import parsers
import metrics as mtr
import projections
import filters


def create_dir_for_results(input):
	base = os.path.basename(input)
	base = (os.path.splitext(base)[0])
	if not os.path.exists('results/%s' % base):
		os.makedirs('results/%s' % base)
	if not os.path.exists('graphs/%s' % base):
		os.makedirs('graphs/%s' % base)
	if not os.path.exists('graphs/%s/tmp' % base):
		os.makedirs('graphs/%s/tmp' % base)
	return base


def clean_tmp():
	dir_name = 'graphs/%s/tmp/' % DIRECTORY
	d = os.listdir(dir_name)

	for item in d:
	    if item.endswith(".p"):
	        os.remove(os.path.join(dir_name, item))


def plot_cc_sizes(G, G_name):
	plt.plot(range(len(list(nx.connected_components(G)))), [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
	plt.yscale('log')
	plt.xlabel('Connected component')
	plt.ylabel('Size of connected component')
	plt.xticks(range(1, len(list(nx.connected_components(G))), 3))
	plt.savefig('results/%s/%s_cc_sizes.png' % (DIRECTORY, G_name))
	plt.close()


def calculate(G_name, func, args, name, mode):
	id_result = G_name + '_' + name + '_' + mode
	if not (kwargs['cache'] and os.path.isfile('graphs/%s/tmp/%s.p' % (DIRECTORY, id_result))):
		LOGGER.info('Computing: %s' % (name))
		G = pickle.load(open('graphs/%s/tmp/%s.p' % (DIRECTORY, G_name), 'rb'))
		start_time = time.time()
		result = func(G, **args)

		pickle.dump(result, open('graphs/%s/tmp/%s.p' % (DIRECTORY, id_result), 'wb'))
		end_time = time.time()
		LOGGER.info('Done: %s in %.2f seconds.' % (name, end_time - start_time))
		del G, result
		gc.collect()
	else:
		LOGGER.info('Using %s.p from cache.' % (id_result))
	return (G_name, id_result, name, mode)


def process_filter(filter):
	filter_args = '_'.join(str(x) for x in filter['args'].values())
	graph_name = 'G_FILT_%s_ARGS_%s_PROJ_%s' % (filter['id'], filter_args, PROJECTION['id'])
	if not (kwargs['cache'] and os.path.isfile('graphs/%s/tmp/%s.p' % (DIRECTORY, graph_name))):
		LOGGER.info('Started filter FILT_%s_ARGS_%s' % (filter['id'], filter_args))
		B = pickle.load(open('graphs/%s/tmp/%s.p' % (DIRECTORY, 'B'), 'rb'))

		# two mode filter
		if filter['two_mode']:
			filter_fn = filters.mapped_filters[filter['id']]
			filter_fn(B, **filter['args'])
			plot_cc_sizes(B, graph_name)
			B = max(nx.connected_component_subgraphs(B), key=len)

		G = project_graph(B)
		mtr.remove_attribute(G, 'bipartite')  # remove leftover attributes

		# one mode filter
		if not filter['two_mode']:
			filter_f = filters.mapped_filters[filter['id']]
			filter_f(G, **filter['args'])
			plot_cc_sizes(G, graph_name)
			G = max(nx.connected_component_subgraphs(G), key=len)

		nx2gt(G, graph_name)
		LOGGER.info('Completed filter FILT_%s_ARGS_%s' % (filter['id'], filter_args))
		del B, G
		gc.collect()
	else:
		LOGGER.info('Using %s.p from cache.' % (graph_name))
	return graph_name


def project_graph(B):
	projection_fn = projections.mapped_projections[PROJECTION['id']]
	nodes_to_project = parsers.get_biparite_node_sets(B, PROJECTION['nodes'])
	return projection_fn(B, nodes_to_project)


def inverse(x):
	return 1 / x


def add_distance(G):
	weight = G.edge_properties["weight"]
	G.ep['distance'] = G.new_ep('double', np.apply_along_axis(inverse, 0, weight.a))


def nx2gt(G, graph_name):
	nx.write_graphml(G, 'graphs/%s/tmp/NX_%s.graphml' % (DIRECTORY, graph_name))
	G = gt.load_graph('graphs/%s/tmp/NX_%s.graphml' % (DIRECTORY, graph_name))
	add_distance(G)
	pickle.dump(G, open('graphs/%s/tmp/%s.p' % (DIRECTORY, graph_name), 'wb'))


def add_tasks(tasks, graph_name):
	if NETWORK_METRICS:
		tasks.extend([(graph_name, mtr.mapped_network_metrics[metric['id']], metric.get('args', {}), mtr.get_metric_title(metric), 'network') for metric in NETWORK_METRICS])
	if NODE_METRICS:
		tasks.extend([(graph_name, mtr.mapped_node_metrics[metric['id']], metric.get('args', {}), mtr.get_metric_title(metric), 'node') for metric in NODE_METRICS])
	if EDGE_METRICS:
		tasks.extend([(graph_name, mtr.mapped_edge_metrics[metric['id']], metric.get('args', {}), mtr.get_metric_title(metric), 'edge') for metric in EDGE_METRICS])


def validate_config(config):
	if not config['relationship'] in parsers.mapped_parsers.keys():
		raise Exception('Incorrect config: relationship not in mapped_parsers.')

	if 'projection' in config:
		if not config['projection']['id'] in projections.mapped_projections.keys():
			raise Exception('Incorrect config: projection id not in mapped_projections.')
		if not config['projection']['nodes'] in ['top', 'bottom']:
			raise Exception('Incorrect config: nodes arg should be top or bottom.')

	if 'filters' in config:
		if not set(x['id'] for x in config['filters']).issubset(set(filters.mapped_filters.keys())):
			raise Exception('Incorrect config: filter id not in mapped_filters.')
		if any(x.get('args') is None for x in config['filters'] if 'args' in x):
			raise Exception('Incorrect config: args is empty.')
		if not all(x['args'].get('nodes') in ['top', 'bottom'] for x in config['filters'] if 'args' in x and 'nodes' in x['args']):
			raise Exception('Incorrect config: nodes arg should be top or bottom.')

	if 'bipartite_metrics' in config and not set(x['id'] for x in config['bipartite_metrics']).issubset(set(mtr.mapped_bipartite_network_metrics.keys())):
		raise Exception('Incorrect config: bipartite metrics id not in mapped_bipartite_network_metrics.')
	if 'bipartite_node_metrics' in config and not set(x['id'] for x in config['bipartite_node_metrics']).issubset(set(mtr.mapped_bipartite_node_metrics.keys())):
		raise Exception('Incorrect config: bipartite node metrics id not in mapped_bipartite_node_metrics.')
	if 'network_metrics' in config and not set(x['id'] for x in config['network_metrics']).issubset(set(mtr.mapped_network_metrics.keys())):
		raise Exception('Incorrect config: network metrics id not in mapped_network_metrics.')
	if 'node_metrics' in config and not set(x['id'] for x in config['node_metrics']).issubset(set(mtr.mapped_node_metrics.keys())):
		raise Exception('Incorrect config: node metrics id not in mapped_node_metrics.')
	if 'edge_metrics' in config and not set(x['id'] for x in config['edge_metrics']).issubset(set(mtr.mapped_edge_metrics.keys())):
		raise Exception('Incorrect config: edge metrics id not in mapped_edge_metrics.')

	pass


def main(**kwargs):
	# append all metrics that have to be computed in the collection
	# at the end of the pipeline we can distribute the computation of these metrics
	# each element is a tuple (graph_fp, metric_fn, metric_args, metric_title, callback)
	tasks = []
	graphs = {}

	# Parse bipartite graph
	graph_name = 'B'
	graphs[graph_name] = {'network': {}, 'node': {}, 'edge': {}}
	if BIPARTITE_METRICS:
		tasks.extend([(graph_name, mtr.mapped_bipartite_network_metrics[metric['id']], metric.get('args', {}), mtr.get_metric_title(metric), 'network') for metric in BIPARTITE_METRICS])
	if BIPARTITE_NODE_METRICS:
		tasks.extend([(graph_name, mtr.mapped_bipartite_node_metrics[metric['id']], metric.get('args', {}), mtr.get_metric_title(metric), 'node') for metric in BIPARTITE_NODE_METRICS])

	if kwargs['cache'] and os.path.isfile('graphs/%s/tmp/%s.p' % (DIRECTORY, graph_name)):
		B = pickle.load(open('graphs/%s/tmp/%s.p' % (DIRECTORY, graph_name), 'rb'))
		LOGGER.info('Using %s.p from cache.' % (graph_name))
	else:
		LOGGER.info('Started parsing bipartite graph.')
		B = parsers.mapped_parsers[RELATIONSHIP](kwargs['input'], kwargs['profile'])
		LOGGER.info('Completed parsing bipartite graph.')

		# We are only interested in lcc of B but still want to export the
		# distribution of connected components so we can asses that there is no
		# other significant component
		plot_cc_sizes(B, graph_name)
		B = max(nx.connected_component_subgraphs(B), key=len)
		pickle.dump(B, open('graphs/%s/tmp/%s.p' % (DIRECTORY, graph_name), 'wb'))

	# baseline unfiltered one-mode graph.
	if PROJECTION:
		graph_name = 'G_BASE_PROJ_%s' % (PROJECTION['id'])
		graphs[graph_name] = {'network': {}, 'node': {}, 'edge': {}}
		add_tasks(tasks, graph_name)

		if not (kwargs['cache'] and os.path.isfile('graphs/%s/tmp/%s.p' % (DIRECTORY, graph_name))):
			LOGGER.info('Started projection bipartite graph.')
			G = project_graph(B)
			mtr.remove_attribute(G, 'bipartite')  # remove leftover attributes
			nx2gt(G, graph_name)
			LOGGER.info('Completed projection bipartite graph.')

		if FILTERS:
			with multiprocessing.Pool() as pool:
				map_result = pool.map_async(process_filter, FILTERS)
				results = map_result.get()
				for graph_name in results:
					graphs[graph_name] = {'network': {}, 'node': {}, 'edge': {}}
					add_tasks(tasks, graph_name)

	# Distribute tasks for computation
	with multiprocessing.Pool() as pool:
		LOGGER.info('Start computing tasks')
		start_time = time.time()
		map_result = pool.starmap_async(calculate, tasks)
		results = map_result.get()
		end_time = time.time()
		LOGGER.info('Done computing all tasks. Total runtime: %.2f seconds.' % (end_time - start_time))

	for G_name, id_result, name, mode in results:
		graphs[G_name][mode][name] = pickle.load(open('graphs/%s/tmp/%s.p' % (DIRECTORY, id_result), 'rb'))

	df = pd.DataFrame.from_dict({k: v['network'] for k, v in graphs.items()}, orient="index")
	df.to_csv('results/%s/%s_%d.csv' % (DIRECTORY, kwargs['output'], int(time.time())), sep=";", index_label='graph_name')

	if kwargs['graphml']:
		for G_name, v in graphs.items():
			G = pickle.load(open('graphs/%s/tmp/%s.p' % (DIRECTORY, G_name), 'rb'))
			if isinstance(G, type(nx.Graph())):
				for name, values in v['node'].items():
					nx.set_node_attributes(G, values, name)
				for name, values in v['edge'].items():
					nx.set_edge_attributes(G, values, name)
				nx.write_graphml(G, 'graphs/%s/%s.graphml' % (DIRECTORY, G_name))
			else:
				for name, values in v['node'].items():
					# Betweenness centraliy returns vp and ep in a tuple
					if type(values) == tuple:
						value = values[1]
						# Convert to python type and get class name
						value_type = type(value.dtype.type(0).item()).__name__
						ep = G.new_ep(value_type)
						for index, vertex in enumerate(G.edges()):
							ep[vertex] = value[index]
						G.ep[name] = ep
						values = values[0]

					# Convert to python type and get class name
					value_type = type(values.dtype.type(0).item()).__name__
					vp = G.new_vp(value_type)
					for index, vertex in enumerate(G.vertices()):
						vp[vertex] = values[index]
					G.vp[name] = vp

				# TODO EDGE PROPERTIES

				G.save('graphs/%s/%s.graphml' % (DIRECTORY, G_name))


if __name__ == '__main__':
	multiprocessing.freeze_support()
	start_time = time.time()
	parser = argparse.ArgumentParser(
		description='Pipeline to apply filters, projection and compute metrics on a social network.')
	parser.add_argument('-i', '--input', dest='input', nargs='?', help='Csv input file path.', type=str, required=True)
	parser.add_argument('-o', '--output', dest='output', nargs='?', help='Output filename for the network metrics.', type=str, required=True)
	parser.add_argument('-p', '--pipeline', dest='config', nargs='?', help='Pipeline config yaml file.', type=str, required=True)
	parser.add_argument('--export-graphml', dest='graphml', help='Export graphs to graphml.', action='store_true')
	parser.add_argument('--cache', dest='cache', help='Used cached pickle graph objects instead of recreating the graphs.', action='store_true')
	parser.add_argument('--profile', dest='profile', nargs='?', help='Csv profile file path. Column Id will be used as index and should contain user Ids', type=str)
	kwargs = vars(parser.parse_args())
	if not os.path.isfile(kwargs['input']):
		raise Exception('Input filepath does not exist.')
	if kwargs['profile'] and not os.path.isfile(kwargs['profile']):
		raise Exception('profile filepath does not exist.')
	# TODO: validate config file
	config = parsers.parse_yaml(kwargs['config'])
	validate_config(config)
	DIRECTORY = create_dir_for_results(kwargs['input'])

	RELATIONSHIP, FILTERS, PROJECTION, BIPARTITE_METRICS, BIPARTITE_NODE_METRICS, NETWORK_METRICS, NODE_METRICS, EDGE_METRICS = config.get('relationship'), config.get('filters'), config.get('projection'), config.get('bipartite_metrics'), config.get('bipartite_node_metrics'), config.get('network_metrics'), config.get('node_metrics'), config.get('edge_metrics')

	LOGGER = multiprocessing.log_to_stderr()
	formatter = logging.Formatter('[%(levelname)s/%(processName)s]\t%(asctime)-15s\t%(message)s', '%Y-%d-%m %H:%M:%S')
	LOGGER.handlers[0].setFormatter(formatter)
	LOGGER.setLevel(logging.INFO)
	LOGGER.info('OpenMP for parallel computation enabled: %s' % gt.openmp_enabled())
	LOGGER.info('Initializing with %d number of threads' % gt.openmp_get_num_threads())
	LOGGER.info('Using config: %s' % config)
	main(**kwargs)
	# if not kwargs['cache']:
	# 	clean_tmp()
	end_time = time.time()
	LOGGER.info('Done with pipeline. Total runtime: %.2f seconds.' % (end_time - start_time))
