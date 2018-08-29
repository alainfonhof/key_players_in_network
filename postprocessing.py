import multiprocessing
import pandas as pd
import graph_tool.all as gt
import os
import re
import argparse


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def join_datasets_on_prop(prop, dir_name, index_label='_graphml_vertex_id'):
	first = True
	df = None
	d = os.listdir('%s/graphs/%s/' % (base_dir, dir_name))
	for g_file in d:
		# only use graph object but exclude B graph since there is only 1 B and we can not compare it to another B graph
		if (g_file.endswith(".graphml") or g_file.endswith(".gt")) and not g_file.startswith('B'):
			print('join %s on graph %s' % (prop, g_file))
			g_path = '%s/graphs/%s/%s' % (base_dir, dir_name, g_file)
			g = gt.load_graph(g_path)
			df_other = pd.DataFrame(index=g.vp[index_label].get_2d_array([0])[0])
			g_file = os.path.splitext(g_file)[0]
			df_other[g_file] = g.vp[prop].a
			if first:
				df = df_other
				first = False
			else:
				df = df.join(df_other)
	df = df.round(5)
	return df


def node_metrics_to_df(g):
	df = pd.DataFrame()
	properties = [x[1] for x in g.properties.keys() if x[0] == 'v']
	for prop in properties:
		print('node metric on prop %s' % prop)
		if g.vp[prop].python_value_type() == str:
			df[prop] = g.vp[prop].get_2d_array([0])[0]
		else:
			df[prop] = g.vp[prop].a
	df = df.round(5)
	return df


def edge_metrics_to_df(g):
	df = pd.DataFrame()
	properties = [x[1] for x in g.properties.keys() if x[0] == 'e']
	for prop in properties:
		print('edge metric on prop %s' % prop)
		if g.ep[prop].python_value_type() == str:
			df[prop] = g.ep[prop].get_2d_array([0])[0]
		else:
			df[prop] = g.ep[prop].a
	df = df.round(5)
	return df


def a(g_file):
	if os.path.isfile('%s/results/%s/node_id_x_node_metrics_%s.csv' % (base_dir, dir_name, os.path.splitext(g_file)[0])):
		return
	if g_file.endswith(".graphml") or g_file.endswith(".gt"):
		g_path = '%s/graphs/%s/%s' % (base_dir, dir_name, g_file)
		print('load %s' % g_file)
		g = gt.load_graph(g_path)
		print('node metric to df')
		df = node_metrics_to_df(g)
		g_file = os.path.splitext(g_file)[0]
		df.to_csv('%s/results/%s/node_id_x_node_metrics_%s.csv' % (base_dir, dir_name, g_file), sep=';', index=False)


def b(prop):
	if os.path.isfile('%s/results/%s/node_x_graphs_%s.csv' % (base_dir, dir_name, prop)):
		return
	print('join df on %s' % prop)
	df = join_datasets_on_prop(prop, dir_name)
	x = df.columns.tolist()
	x.sort(key=natural_keys)
	df.reindex(x, axis="columns").to_csv('%s/results/%s/node_x_graphs_%s.csv' % (base_dir, dir_name, prop), sep=';')


if __name__ == '__main__':
	# input = graphml graphs
	# parse to gt graph objects
	# output = df node_id x node metrics per graph
	# output = df edge_id x edge metrics per graph
	# output = df node_id x graph per property
	# set either dir_name or g_name
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', dest='input', nargs='?', help='Csv input file path.', type=str, required=True)
	kwargs = vars(parser.parse_args())
	dir_name = kwargs['input']
	base_dir = '/data'
	properties = ['betweenness_centrality', 'closeness_centrality', 'clustering', 'degree_centrality', 'pagerank', 'eigenvector', 'average_distance', 'average_neighbor_degree', 'average_weighted_degree', 'weighted_degree']

	d = os.listdir('%s/graphs/%s/' % (base_dir, dir_name))

	with multiprocessing.Pool() as pool:
		pool.map(a, d)
	with multiprocessing.Pool() as pool:
		pool.map(b, properties)
