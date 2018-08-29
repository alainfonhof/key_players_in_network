import multiprocessing
import networkx as nx
import os
import pickle
from pathlib import Path


def a(g_file):
	r_path = Path('results/%s/%s_rc.p' % (dir_name, g_file))
	if g_file.startswith("NX") and not r_path.exists():
		g_path = 'graphs/%s/tmp/%s' % (dir_name, g_file)
		print('loading: %s' % g_file)
		G = nx.read_graphml(g_path)
		print('computing: %s' % g_file)
		rc = nx.rich_club_coefficient(G)
		print('finished computing: %s' % g_file)
		pickle.dump(rc, open('results/%s/%s_rc.p' % (dir_name, g_file), 'wb'))


if __name__ == '__main__':
	# set either dir_name
	dir_name = 'revolver_bipartite'
	d = os.listdir('graphs/%s/tmp/' % dir_name)
	with multiprocessing.Pool() as pool:
		pool.map(a, d)
