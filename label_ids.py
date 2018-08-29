import pandas as pd
import graph_tool.all as gt

csv_file = ''
graph_file = ''
df = pd.read_csv(file, sep=';')
G = gt.load_graph(graph_file)

label_vp = G.vp['Label']
index_vp = G.vp['_graphml_vertex_id']
for i, row in df.iterrows():
    name = row['Name']
    for vertex in G.vertices():
        if label_vp[vertex] == name:
            df.at[i, 'Id'] = index_vp[vertex]
            break

df.to_csv('MK_ResultsPIM.csv', sep=';', index=False)
