import networkx as nx
from networkx.algorithms import bipartite


def projected_graph(B, nodes, multigraph=False):
	"""Returns the projection of B onto one of its node sets.

	Returns the graph G that is the projection of the bipartite graph B
	onto the specified nodes. They retain their attributes and are connected
	in G if they have a common neighbor in B.

	Parameters
	----------
	B : NetworkX graph
	  The input graph should be bipartite.

	nodes : list or iterable
	  Nodes to project onto (the "bottom" nodes).

	multigraph: bool (default=False)
	   If True return a multigraph where the multiple edges represent multiple
	   shared neighbors.  They edge key in the multigraph is assigned to the
	   label of the neighbor.

	Returns
	-------
	Graph : NetworkX graph or multigraph
	   A graph that is the projection onto the given nodes.

	Examples
	--------
	>>> from networkx.algorithms import bipartite
	>>> B = nx.path_graph(4)
	>>> G = bipartite.projected_graph(B, [1, 3])
	>>> list(G)
	[1, 3]
	>>> list(G.edges())
	[(1, 3)]

	If nodes `a`, and `b` are connected through both nodes 1 and 2 then
	building a multigraph results in two edges in the projection onto
	[`a`, `b`]:

	>>> B = nx.Graph()
	>>> B.add_edges_from([('a', 1), ('b', 1), ('a', 2), ('b', 2)])
	>>> G = bipartite.projected_graph(B, ['a', 'b'], multigraph=True)
	>>> print([sorted((u, v)) for u, v in G.edges()])
	[['a', 'b'], ['a', 'b']]

	Notes
	-----
	No attempt is made to verify that the input graph B is bipartite.
	Returns a simple graph that is the projection of the bipartite graph B
	onto the set of nodes given in list nodes.  If multigraph=True then
	a multigraph is returned with an edge for every shared neighbor.

	Directed graphs are allowed as input.  The output will also then
	be a directed graph with edges if there is a directed path between
	the nodes.

	The graph and node properties are (shallow) copied to the projected graph.

	See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
	for further details on how bipartite graphs are handled in NetworkX.

	See Also
	--------
	is_bipartite,
	is_bipartite_node_set,
	sets,
	weighted_projected_graph,
	collaboration_weighted_projected_graph,
	overlap_weighted_projected_graph,
	generic_weighted_projected_graph
	"""
	if B.is_multigraph():
		raise nx.NetworkXError("not defined for multigraphs")
	if B.is_directed():
		directed = True
		if multigraph:
			G = nx.MultiDiGraph()
		else:
			G = nx.DiGraph()
	else:
		directed = False
		if multigraph:
			G = nx.MultiGraph()
		else:
			G = nx.Graph()
	G.graph.update(B.graph)
	G.add_nodes_from((n, {**B.nodes[n], **{'start': min([x[2] for x in B.edges(n, data='start')])}}) for n in nodes)
	for u in nodes:
		nbrs2 = set(v for nbr in B[u] for v in B[nbr] if v != u)
		if multigraph:
			for n in nbrs2:
				if directed:
					links = set(B[u]) & set(B.pred[n])
				else:
					links = set(B[u]) & set(B[n])
				for l in links:
					if not G.has_edge(u, n, l):
						G.add_edge(u, n, key=l)
		else:
			G.add_edges_from((u, n, {'weight': 1, 'start': start_conversation(B.edges(u, data='start'), B.edges(n, data='start'))}) for n in nbrs2)
	return G


# We need the weight to be type float instead of int so we slightly adjust the
# networkx weighted_projected_graph function.
def weighted_projected_graph(B, nodes, ratio=False):
	"""Returns a weighted projection of B onto one of its node sets.

	The weighted projected graph is the projection of the bipartite
	network B onto the specified nodes with weights representing the
	number of shared neighbors or the ratio between actual shared
	neighbors and possible shared neighbors if ``ratio is True`` [1]_.
	The nodes retain their attributes and are connected in the resulting
	graph if they have an edge to a common node in the original graph.

	Parameters
	----------
	B : NetworkX graph
		The input graph should be bipartite.

	nodes : list or iterable
		Nodes to project onto (the "bottom" nodes).

	ratio: Bool (default=False)
		If True, edge weight is the ratio between actual shared neighbors
		and possible shared neighbors. If False, edges weight is the number
		of shared neighbors.

	Returns
	-------
	Graph : NetworkX graph
	   A graph that is the projection onto the given nodes.

	Examples
	--------
	>>> from networkx.algorithms import bipartite
	>>> B = nx.path_graph(4)
	>>> G = bipartite.weighted_projected_graph(B, [1, 3])
	>>> list(G)
	[1, 3]
	>>> list(G.edges(data=True))
	[(1, 3, {'weight': 1})]
	>>> G = bipartite.weighted_projected_graph(B, [1, 3], ratio=True)
	>>> list(G.edges(data=True))
	[(1, 3, {'weight': 0.5})]

	Notes
	-----
	No attempt is made to verify that the input graph B is bipartite.
	The graph and node properties are (shallow) copied to the projected graph.

	See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
	for further details on how bipartite graphs are handled in NetworkX.

	See Also
	--------
	is_bipartite,
	is_bipartite_node_set,
	sets,
	collaboration_weighted_projected_graph,
	overlap_weighted_projected_graph,
	generic_weighted_projected_graph
	projected_graph

	References
	----------
	.. [1] Borgatti, S.P. and Halgin, D. In press. "Analyzing Affiliation
		Networks". In Carrington, P. and Scott, J. (eds) The Sage Handbook
		of Social Network Analysis. Sage Publications.
	"""
	if B.is_directed():
		pred = B.pred
		G = nx.DiGraph()
	else:
		pred = B.adj
		G = nx.Graph()
	G.graph.update(B.graph)
	# print([(n, {**B.nodes[n], **{'start': min([x[2] for x in B.edges(n, data='start')])}}) for n in nodes])
	G.add_nodes_from((n, {**B.nodes[n], **{'start': min([x[2] for x in B.edges(n, data='start')])}}) for n in nodes)
	n_top = float(len(B) - len(nodes))
	for u in nodes:
		unbrs = set(B[u])
		nbrs2 = set((n for nbr in unbrs for n in B[nbr])) - set([u])
		for v in nbrs2:
			vnbrs = set(pred[v])
			common = unbrs & vnbrs
			if not ratio:
				weight = float(len(common))
			else:
				weight = float(len(common) / n_top)
			G.add_edge(u, v, weight=weight, start=start_conversation(B.edges(u, data='start'), B.edges(v, data='start')))
	return G


def collaboration_weighted_projected_graph(B, nodes):
	"""Newman's weighted projection of B onto one of its node sets.

	The collaboration weighted projection is the projection of the
	bipartite network B onto the specified nodes with weights assigned
	using Newman's collaboration model [1]_:

	.. math::

		w_{u, v} = \sum_k \frac{\delta_{u}^{k} \delta_{v}^{k}}{d_k - 1}

	where `u` and `v` are nodes from the bottom bipartite node set,
	and `k` is a node of the top node set.
	The value `d_k` is the degree of node `k` in the bipartite
	network and `\delta_{u}^{k}` is 1 if node `u` is
	linked to node `k` in the original bipartite graph or 0 otherwise.

	The nodes retain their attributes and are connected in the resulting
	graph if have an edge to a common node in the original bipartite
	graph.

	Parameters
	----------
	B : NetworkX graph
	  The input graph should be bipartite.

	nodes : list or iterable
	  Nodes to project onto (the "bottom" nodes).

	Returns
	-------
	Graph : NetworkX graph
	   A graph that is the projection onto the given nodes.

	Examples
	--------
	>>> from networkx.algorithms import bipartite
	>>> B = nx.path_graph(5)
	>>> B.add_edge(1, 5)
	>>> G = bipartite.collaboration_weighted_projected_graph(B, [0, 2, 4, 5])
	>>> list(G)
	[0, 2, 4, 5]
	>>> for edge in G.edges(data=True): print(edge)
	...
	(0, 2, {'weight': 0.5})
	(0, 5, {'weight': 0.5})
	(2, 4, {'weight': 1.0})
	(2, 5, {'weight': 0.5})

	Notes
	-----
	No attempt is made to verify that the input graph B is bipartite.
	The graph and node properties are (shallow) copied to the projected graph.

	See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
	for further details on how bipartite graphs are handled in NetworkX.

	See Also
	--------
	is_bipartite,
	is_bipartite_node_set,
	sets,
	weighted_projected_graph,
	overlap_weighted_projected_graph,
	generic_weighted_projected_graph,
	projected_graph

	References
	----------
	.. [1] Scientific collaboration networks: II.
		Shortest paths, weighted networks, and centrality,
		M. E. J. Newman, Phys. Rev. E 64, 016132 (2001).
	"""
	if B.is_directed():
		pred = B.pred
		G = nx.DiGraph()
	else:
		pred = B.adj
		G = nx.Graph()
	G.graph.update(B.graph)
	G.add_nodes_from((n, {**B.nodes[n], **{'start': min([x[2] for x in B.edges(n, data='start')])}}) for n in nodes)
	for u in nodes:
		unbrs = set(B[u])
		nbrs2 = set(n for nbr in unbrs for n in B[nbr] if n != u)
		for v in nbrs2:
			vnbrs = set(pred[v])
			common_degree = (len(B[n]) for n in unbrs & vnbrs)
			weight = sum(1.0 / (deg - 1) for deg in common_degree if deg > 1)
			G.add_edge(u, v, weight=weight, start=start_conversation(B.edges(u, data='start'), B.edges(v, data='start')))
	return G


def start_conversation(adj_u, adj_v):
	u_starts = [x[2] for x in adj_u]
	v_starts = [x[2] for x in adj_v]
	return max(min(u_starts), min(v_starts))


mapped_projections = {
	1: projected_graph,
	2: weighted_projected_graph,
	3: collaboration_weighted_projected_graph
}
