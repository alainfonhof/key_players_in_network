import networkx as nx
from networkx.algorithms import bipartite
import graph_tool.all as gt
import numpy as np

import parsers


#######
# Bipartite network metrics
#######
def bipartite_density(B, nodes=None):
	"""
	Parameters
	----------
	nodes : string ("top"|"bottom")
		Select either top or bottom nodes
	"""
	node_dict = parsers.get_biparite_node_sets(B)
	if nodes is None:
		return nx.number_of_edges(B) / (len(node_dict['top']) * len(node_dict['bottom']))
	else:
		return bipartite.density(B, node_dict[nodes])


def bipartite_average_degree(B, nodes=None):
	"""
	Parameters
	----------
	nodes : string ("top"|"bottom")
		Select either top or bottom nodes
	"""
	node_dict = parsers.get_biparite_node_sets(B)
	m = nx.number_of_edges(B)
	if nodes is None:
		return (2 * float(m) / (float(len(node_dict['top'])) + float(len(node_dict['bottom']))))
	else:
		return (float(m) / float(len(node_dict[nodes])))


def bipartite_average_clustering(B, nodes=None, mode='dot'):
	"""
	Parameters
	----------
	nodes : string ("top"|"bottom")
		Select either top or bottom nodes
	"""
	if nodes is None:
		return bipartite.average_clustering(B, mode=mode)
	else:
		node_dict = parsers.get_biparite_node_sets(B)
		return bipartite.average_clustering(B, nodes=node_dict[nodes], mode=mode)


def bipartite_number_of_nodes(B, nodes=None):
	"""
	Parameters
	----------
	nodes : string ("top"|"bottom")
		Select either top or bottom nodes
	"""
	if nodes is None:
		return nx.number_of_nodes(B)
	else:
		node_dict = parsers.get_biparite_node_sets(B)
		return len(node_dict[nodes])


#######
# Bipartite node metrics
#######
def bipartite_degree_centrality(B, nodes):
	node_dict = parsers.get_biparite_node_sets(B)
	return bipartite.degree_centrality(B, nodes=node_dict[nodes])


def bipartite_closeness_centrality(B, nodes):
	node_dict = parsers.get_biparite_node_sets(B)
	return bipartite.closeness_centrality(B, nodes=node_dict[nodes])


def bipartite_betweenness_centrality(B, nodes):
	node_dict = parsers.get_biparite_node_sets(B)
	return bipartite.betweenness_centrality(B, nodes=node_dict[nodes])


def bipartite_average_neighbor_degree(B):
	avg = {}
	for n in B:
		avg[n] = sum(B.degree[nbr] for nbr in B.neighbors(n)) / B.degree[n]
	return avg

#######
# Network metrics
#######
# def average_degree(G):
# 	""" Average degree of all nodes in G.

# 	Parameters
# 	----------
# 	G : nx.Graph
# 		Graph

# 	Returns
# 	-------
# 	average_degree : float
# 		Average degree of G
# 	"""
# 	nnodes = G.number_of_nodes()
# 	if len(G) > 0:
# 		if G.is_directed():
# 			return (sum(d for n, d in G.in_degree()) / float(nnodes), sum(d for n, d in G.out_degree()) / float(nnodes))
# 		else:
# 			s = sum(dict(G.degree()).values())
# 			return (float(s) / float(nnodes))


def average_weighted_degree(G):
	weight = G.ep['weight']
	deg = G.degree_property_map('total', weight)
	return gt.vertex_average(G, deg)[0]


def number_of_nodes(G):
	return G.num_vertices()


def number_of_edges(G):
	return G.num_edges()


def density(G):
	n = G.num_vertices()
	m = G.num_edges()
	if m == 0 or n <= 1:
		d = 0.0
	else:
		if G.is_directed():
			d = m / float(n * (n - 1))
		else:
			d = m * 2.0 / float(n * (n - 1))
	return d


# distance
def average_shortest_path_length(G):
	n = G.num_vertices()
	eprop = G.edge_properties['distance']
	return np.sum(gt.shortest_distance(G, weights=eprop).get_2d_array(range(n))) / (n * (n - 1))


# distance
def pseudo_diameter(G):
	eprop = G.edge_properties['distance']
	nn = min(G.num_vertices(), 10)
	dia = [gt.pseudo_diameter(G, source=n, weights=eprop)[0] for n in range(nn)]
	return max(dia)


def weighted_degree_assortativity_coefficient(G):
	weight = G.ep['weight']
	deg = G.degree_property_map('total', weight)
	return gt.scalar_assortativity(G, deg)[0]


def average_clustering(G):
	return gt.global_clustering(G)[0]


#######
# Node metrics
#######
def local_clustering(G):
	clust = gt.local_clustering(G)
	return clust.a


def degree_centrality(G):
	s = 1.0 / (G.num_vertices() - 1.0)
	deg = G.new_vertex_property('float')
	for vertex in G.vertices():
		deg[vertex] = s * (vertex.in_degree() + vertex.out_degree())
	return deg.a


# distance
def closeness_centrality(G):
	eprop = G.edge_properties['distance']
	closeness = gt.closeness(G, weight=eprop)
	return closeness.a


# distance
def betweenness_centrality(G):
	eprop = G.edge_properties['distance']
	vp, ep = gt.betweenness(G, weight=eprop)
	return (vp.a, ep.a)


def pagerank(G):
	eprop = G.edge_properties['weight']
	pagerank = gt.pagerank(G, weight=eprop)
	return pagerank.a


def eigenvector(G):
	eprop = G.edge_properties['weight']
	eigenvalue, eigenvector = gt.eigenvector(G, weight=eprop)
	return eigenvector.a


def avg_weight(G):
	weight = G.ep['weight']
	avg_w = G.new_vp('double')
	w_deg = G.degree_property_map('total', weight)
	for v in G.vertices():
		avg_w[v] = w_deg[v] / len(G.get_in_neighbors(v))
	return avg_w.a


def avg_distance(G):
	weight = G.ep['distance']
	avg_d = G.new_vp('double')
	w_deg = G.degree_property_map('total', weight)
	for v in G.vertices():
		avg_d[v] = w_deg[v] / len(G.get_in_neighbors(v))
	return avg_d.a


def weighted_degree(G):
	weight = G.ep['weight']
	w_deg = G.degree_property_map('total', weight)
	return G.new_vp('double', w_deg.a).a

# DEPRECEATED - should be computed in different analysis because
# to much information is lost when dropping the block state
# def community(G):
# 	eprop = G.edge_properties['weight']
# 	state = gt.minimize_blockmodel_dl(G, recs={'eweight': eprop.a})
# 	return state.b.a


def average_neighbor_degree(G):
	weight = G.ep['weight']
	deg = G.degree_property_map('total', weight)
	avg_neighbor_degree = G.new_vp('double')
	for v in G.vertices():
		avg_neighbor_degree[v] = sum([deg[w] for w in v.all_neighbors()]) / len(G.get_in_neighbors(v))
	return avg_neighbor_degree.a


#######
# Helpers
#######
def remove_attribute(G, metrics):
	if isinstance(metrics, str):
		for node in G.nodes:
			del G.nodes[node][metrics]
	else:
		for metric in metrics:
			metric_title = get_metric_title(metric)
			for node in G.nodes:
				del G.nodes[node][metric_title]


def get_metric_title(metric):
	args = [str(v) for k, v in metric.get('args', {}).items() if k != 'weight']
	if args:
		return metric['id'] + '_' + '_'.join(args)
	else:
		return metric['id']


mapped_bipartite_network_metrics = {
	'nodes': bipartite_number_of_nodes,
	'edges': nx.number_of_edges,
	'density': bipartite_density,
	'average_degree': bipartite_average_degree,
	'average_clustering': bipartite_average_clustering
}

mapped_bipartite_node_metrics = {
	'clustering': bipartite.clustering,
	'degree_centrality': bipartite_degree_centrality,
	'closeness_centrality': bipartite_closeness_centrality,
	'betweenness_centrality': bipartite_betweenness_centrality,
	'average_neighbor_degree': nx.average_neighbor_degree
}

mapped_network_metrics = {
	'nodes': number_of_nodes,
	'edges': number_of_edges,
	'average_degree': average_weighted_degree,
	'density': density,
	'average_shortest_path_length': average_shortest_path_length,
	'diameter': pseudo_diameter,
	'degree_assortativity_coefficient': weighted_degree_assortativity_coefficient,
	'average_clustering': average_clustering,
	# 'ratio_removed_edges': ratio_removed_edges,
	# 'delta_part_sim': partition_similarity_measure
}

mapped_node_metrics = {
	'average_neighbor_degree': average_neighbor_degree,
	'clustering': local_clustering,
	'degree_centrality': degree_centrality,
	'closeness_centrality': closeness_centrality,
	'betweenness_centrality': betweenness_centrality,
	'pagerank': pagerank,
	'eigenvector': eigenvector,
	'average_weighted_degree': avg_weight,
	'average_distance': avg_distance,
	'weighted_degree': weighted_degree
}

mapped_edge_metrics = {}
