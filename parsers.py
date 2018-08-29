import yaml
from datetime import datetime

import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import graph_tool.all as gt


def parse_yaml(yml):
	with open(yml) as stream:
		try:
			return yaml.load(stream)
		except yaml.YAMLError as exc:
			raise Exception(exc)


def parse_publication_date(date_string):
	try:
		date = datetime.strptime(date_string, '%m/%d/%Y %H:%M:%S')
		return date.strftime('%Y-%m-%d %H:%M:%S')
	except Exception as e:
		raise(e)


def get_biparite_node_sets(B, get_set=None):
	if not bipartite.is_bipartite(B):
		raise Exception('Input graph should be bipartite.')
	# Using the bipartite node attribute, we can easily get the two node sets:
	top_nodes = set(n for n, d in B.nodes(data=True) if d['bipartite'] == 0)
	bottom_nodes = set(B) - top_nodes
	nodes = {'top': list(top_nodes), 'bottom': list(bottom_nodes)}
	if not bipartite.is_bipartite_node_set(B, bottom_nodes) or not bipartite.is_bipartite_node_set(B, top_nodes):
		raise Exception('All nodes belonging to the same set should be in the same column.')
	if get_set:
		return nodes[get_set]
	else:
		return nodes


# TODO column 2..n als extra data parsen.
def parse_bipartite(input, profile=None):
	""" Parse a csv file to a bipartite graph.
	The csv file should have no header row.
	The csv file should have a source and target on the first and second column.
	The csv file should have a timestamp on the third column.

	Parameters
	----------
	input : string
		String of csv file path.

	Returns
	-------
	B : nx.Graph
		Undirected bipartitle networkx Graph.
	"""
	B = nx.Graph()
	with open(input, 'r') as f:
		for line in f.readlines():
			data = line.rstrip('\n').split(',')
			B.add_node('TOPIC_' + data[0], bipartite=0)  # topic
			B.add_node(data[1], bipartite=1)  # user
			B.add_edge('TOPIC_' + data[0], data[1], weight=1.0, start=datetime.fromtimestamp(float(data[2])).strftime('%Y-%m-%d %H:%M:%S'))
	return B


def parse_toolbox_public(input, profile=None):
	""" Parse a csv file to a bipartite graph.
	Skips header row.

	Parameters
	----------
	input : string
		String of csv file path.

	Returns
	-------
	B : nx.Graph
		Undirected bipartitle networkx Graph.
	"""
	B = nx.Graph()
	with open(input, 'r', encoding="utf8") as f:
		next(f)  # skip header
		for line in f.readlines():
			data = line.rstrip('\n').split(';')
			B.add_node(data[1], bipartite=1, Label=data[2])  # user
			B.add_node('TOPIC_' + data[3], bipartite=0, Label=data[4], environment_id=data[5], environment_name=data[6], environment_parent_environment_id=data[7], environment_parent_environment_name=data[8])  # topic
			B.add_edge(data[1], 'TOPIC_' + data[3], start=parse_publication_date(data[0]), weight=1.0)
	if profile:
		df = pd.read_csv(profile, sep=";", dtype={'Id': str})
		df.set_index('Id', inplace=True)
		x = df.to_dict()
		for k, v in x.items():
			nx.set_node_attributes(B, v, k)
	return B


def parse_toolbox_private(input):
	return gt.load_graph_from_csv(input, ecols=(1, 3), eprop_names=['EventDate', 'Author.Name', 'Receiver.Name'], skip_first=True, csv_options={"delimiter": ";"})


mapped_parsers = {
	1: parse_bipartite,
	2: parse_toolbox_public,
	3: parse_toolbox_private
}
