import Pycluster
import numpy
import math

def recursive_parsing(n, tree , out):
	if n>=0:
		if len(out)==0:
			return [n]
		else:
			return out.append(n)
	else:
		n1=recursive_parsing(tree[-1*n-1].right,tree,out)
		
		n2=recursive_parsing(tree[-1*n-1].left,tree,out)
		
		out=out+n1
		out=out+n2
		return out


def cluster_frechet(dist_file, lbl):
	dist=numpy.loadtxt(dist_file)
	parent=dict()
	C=dict()
	
	if len(dist.shape)!=2:
		path_cnt=math.sqrt(len(dist))
		dist=numpy.reshape(dist, [path_cnt,path_cnt])	

	
	if lbl is not None and lbl!=-1:
		dist=numpy.delete(dist,lbl,0)	#delete the row and the column of lbl from dists
		dist=numpy.delete(dist,lbl,1)

	tree=Pycluster.treecluster(distancematrix=dist)

	#return tree
	#for cluster in tree:
	
	for idx, cluster in enumerate(tree):
		parent[cluster.right]=-1*idx-1
		parent[cluster.left]=-1*idx-1
		clusters=[]
	
		clusters=clusters + (recursive_parsing(cluster.left,tree,[]))
		clusters= clusters + (recursive_parsing(cluster.right,tree,[]))
		#print cluster.right
	
	
		#print sorted(clusters)
		C[idx]=clusters
	


	return [tree, C, parent]

	
	



	
	

