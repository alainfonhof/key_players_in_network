import pandas.api.types as types
import pandas as pd
import graph_tool.all as gt
import os


def add_attributes(g_file, df):
	g = gt.load_graph(g_file)
	index_vp = g.vp['_graphml_vertex_id']
	props = [x[1] for x in g.properties.keys()]

	for column in df:
		if column in props:
			print(column, ' is already a property in the graph and will not be added.')
			continue
		if types.is_string_dtype(df[column]):
			new_vp = g.new_vp('string')
		elif types.is_float_dtype(df[column]):
			new_vp = g.new_vp('float')
		elif types.is_bool_dtype(df[column]):
			new_vp = g.new_vp('bool')
		elif types.is_int64_dtype(df[column]):
			new_vp = g.new_vp('int64_t')
		elif types.is_integer_dtype(df[column]):
			new_vp = g.new_vp('int')
		elif types.is_object_dtype(df[column]):
			new_vp = g.new_vp('object')
		else:
			raise Exception('Unkown dtype')

		for i, row in df.iterrows():
			for vertex in g.vertices():
				if index_vp[vertex] == i:
					new_vp[vertex] = df.loc[i, column]
					break
		g.vp[column] = new_vp
	g.save(g_file)


if __name__ == '__main__':
	# set either dir_name or g_name
	dir_name = ''
	g_file = ''
	csv_file = ''

	assert g_file == '' or dir_name == ''
	print('read_csv')
	df = pd.read_csv(csv_file, sep=';', index_col='Id')
	df.index = df.index.astype(str)

	if dir_name:
		d = os.listdir('graphs/%s/' % dir_name)
		for item in d:
			if item.endswith(".graphml") or item.endswith(".gt"):
				g_file = os.path.join(dir_name, item)
				print('loading %s' % g_file)
				add_attributes(g_file, df)
	else:
		g = gt.load_graph(g_file)
		add_attributes(g, df)
