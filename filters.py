import parsers


def remove_biggest_linker(B, nodes, n):
	""" Removes biggest linker in a bipartite graph and from the node list.

	Parameters
	----------
	B : nx.Graph
		Bipartite graph
	nodes : list
		List with top or bottom nodes to remove biggest linker from.

	Returns
	-------
	nodes : list
		The new node list with the biggest linker removed.
	biggest_linker : dict
		Biggest linker in nodes.
	"""
	nodes = parsers.get_biparite_node_sets(B, nodes)
	for _ in range(n):
		biggest_linker = max(dict(B.degree(nodes)), key=dict(B.degree(nodes)).get)
		B.remove_node(biggest_linker)
		# nodes.remove(biggest_linker)
	# return nodes


def remove_minimal_weight_edges(G, n):
	""" Remove edges that have a lower weight than n.
	Nodes in the graph should have weight data attribute.

	Parameters
	----------
	G : nx.Graph
		Graph
	n : int or float
		Edges with a weight lower than n will be removed.
	"""
	G.remove_edges_from([e for e in G.edges.data('weight') if e[2] < n])


def remove_user_minimal_degree(B, n):
	""" Remove users that have a lower or equal degree as n.
	Users are identified by bipartite == 1.

	Parameters
	----------
	B : nx.Graph
		Graph
	n : int or float
		Edges with a weight lower than n will be removed.
	"""
	B.remove_nodes_from([node[0] for node in B.nodes.data('bipartite') if B.degree[node[0]] <= n and node[1] == 1])


def filter_on_attribute(B, attr, value):
	B.remove_nodes_from([n for n, data in B.nodes(data=True) if data.get(attr) != value])


mapped_filters = {
	1: remove_biggest_linker,
	2: remove_minimal_weight_edges,
	3: remove_user_minimal_degree,
	4: filter_on_attribute
}
