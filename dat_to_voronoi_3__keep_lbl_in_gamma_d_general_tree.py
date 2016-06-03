#!/usr/bin/env python2.7

import numpy 
import frechet
import cluster_frechet_hierarchical
import scipy.spatial as sp
import random
import math
import sys
import os.path
import pdb
import time
#import matplotlib.pyplot as plt
from scipy.misc import imread
#######################################################################


class CTreeNode(object):
    distance = 0
    id=-1
    nodes = []
    

    
    def __init__(self, distance, nodes, id):
        self.distance = distance
        self.nodes = nodes
	self.id=id
        

#########################################################################

def dynamics_lin2(D,z,rng,dynamics_noise): 	#a distance-weighted average of next points
	
	nrm=0
	prediction=0
	for i in range(len(D)):		
		#find distance to z
		dist=math.sqrt((D[i,0]-z[0])**2 + (D[i,1]-z[1])**2)
		alpha=1/(dist+0.000000001)
		nrm+=alpha
		prediction+=alpha*D[i,2]

	if nrm!=0: prediction/=nrm
	prediction+=dynamics_noise*(random.random()*2-1)
	return prediction


#######################################################################


'''

Observation function
Takes in the observation and the estimated position
Returns a value inversly proportional to the distance between the two points
Does that depend on the class at all??



returns a value between 0 and 1 (usually)
'''
def update_particle(o,z,c,rng):
	#to update the class, find the distance from the current estimate to the observation
	d=math.sqrt((o[0]-z[0])**2+(o[1]-z[1])**2)
	d=(d+0.000001)/rng
	if d<0.0001:
		d=0.0001
		
	return abs(math.log(min(d,1))) 



def update_particle_c(o_c,p_c,trj_cnt,Parent,rng, tree):
	#update knowing expected class c
	
	p1=o_c
	p2=p_c

	d=class_dist2(p1,p2,Parent, trj_cnt,tree)
	d=(d+0.000001)/rng
	if d<0.0001:
		d=0.0001

	return abs(math.log(min(d,1)))

#######################################################################


'''
difference in birth index to the shared father
'''
def class_dist1(p1,p2,Parent, trj_cnt,tree):
	if p1>=0: 	#observation class positive
		if p1<trj_cnt:		# and an a single trajectory
			p1=Parent[p1]	# take its parent					
		else:
			p1=-1*(p1-trj_cnt)-1;
	if p2>=0:
		if p2<trj_cnt:		
			p2=Parent[p2]
		else:
			p2=-1*(p2-trj_cnt)-1;

	#assert p1<0 and p2<0

	#find distance between classes
	if p1<p2:
		min_p=p2
	else:
		min_p=p1

	while(p1!=p2):
		#find the one with a smaller parent (negative)
		if p1>p2: p1=Parent[p1]
		else: p2=Parent[p2]
	
	#now we reached a shared parent p1=p2, compute the distance
	return (tree[-1-p1].distance-tree[-1-min_p].distance)
		



################################################################################

'''
A modified version of the metric, returning birth index of the shared father
'''
def class_dist2(p1,p2,Parent, trj_cnt,tree):
	#find distance between classes
	if p1>=0: 	#observation class positive
		if p1<trj_cnt:		# and an a single trajectory
			p1=Parent[p1]	# take its parent					
		else:
			p1=-1*(p1-trj_cnt)-1;
	if p2>=0:
		if p2<trj_cnt:		
			p2=Parent[p2]
		else:
			p2=-1*(p2-trj_cnt)-1;

	#assert p1<0 and p2<0

	while(p1!=p2):
		#find the one with a smaller parent (negative)
		if p1>p2: p1=Parent[p1]
		else: p2=Parent[p2]
	
	#now we reached a shared parent p1=p2, compute the distance
	return tree[-1-p1].distance
		



################################################################################

'''
given an observation point and a set of points of k trajectories of some class, it computes the average probability of the k closest points as computed by update_particle(o,z).
This will favor a single trajectory class over a multiple trajectory class if it was the closest
'''
def probability_of_class(o,k,kd, rng):
	d=kd.query(o,k) #closest k points to o
	#to update the class, find the distance from the current estimate to the observation
	dis=0
	for i in range(k):
		try:
			d[0].shape #if d[0].shape==():
		except:
			dis+=d[i]#**2
		else:
			dis+=d[0][i]#**2
	
	#return 0.1*abs(math.log(min((dis+0.0001)/k,1))) # 0.1 log of average distance to class points

	dis=(dis+0.0001)/(k*rng)   ############################################################################!!!!!!!!######
	if dis<0.001:
		dis=0.001
		
	return abs(math.log(min(dis,1))) 




################################################################################

'''

given a coarse observation (made by a mean m and variance v), samples a number of points around the observation, and computes the sum of probabilities of the points under all classes, then returns the most probable class for that observation

'''

def most_probable_class(m,v, cls_cnt, kdds,rng,lbl,trj_cnt,C):
	#sample n points
	#pdb.set_trace()
	n=10
	Pts=dict()
	for i in range(n):
		Pts[i]=sample_point(m,v)

	max_Pr=0
	best_class=-1

	
	#compute probabilities under every class
	
	for c in range(cls_cnt):
		if c<trj_cnt: continue 			##################################################################
		if c==lbl:continue
		Pr=0.0
		#get the points of class c into D
		if c < trj_cnt:
			k=1
		else:
			k=len(C[c-trj_cnt])	# number of components (trajectories)
			
							
		#get the number of trajectories in c
		for i in range(n):			
			Pr+=probability_of_class(Pts[i],k,kdds[c],rng)

		if Pr>max_Pr:
			max_Pr=Pr
			best_class=c

	return best_class
		
'''

finds the region closest to the obesrvation by comparing it to the center points, and returns the class for that region at all levels


def observation_to_class(o,C,D,I,tree,e,z):
	#apply a kd tree algorithm on the points, keep indices of classes	
	kd=sp.cKDTree(D)
	[d,i]=kd.query(z)
	num_trajs=1+max(I)

		
	closest_point=D[i]
	return prob_point(closest_point,e/2,o)



def prob_point(m,s,o):
	prob=scipy.stats.multivariate_normal(mean=m, cov=s**2).pdf(o)
	return prob

'''

#######################################################################

'''

Point sampling procedure

'''
def sample_point(m,s):
	point=[random.gauss(m[0], s),random.gauss(m[1], s)]   
	return point



#######################################################################

'''

rebuild tree from the lowest level particles

'''

def rebuild_tree(trj_cnt,P_c,tree,N):
	P=[0]*trj_cnt 	#probability at the lowest level
	Ixs_c=[[] for i in range(trj_cnt)]
	#compute probabilities of classes
	for c in range(trj_cnt):	#only for classes in the lowest level
		I=numpy.where(numpy.array(P_c)==c) 	#indices of particles of class c
		I = [map(int, x) for x in I][0]
		Ixs_c[c]=I
		P[c]=(len(I)+0.0)/N 	#number of particles in c, divided by the number of particles at the lowest level

		#assert P[c]<=1

	y=N
	#compute probabilities of rest of the tree
	for l in range(len(tree)):
		#compute probability at level l+1 and sample particles
		c=l+trj_cnt
	
		n=0
		for nd in tree[l].nodes:
			if n>=0: 
				len_nd=len(Ixs_c[nd])
			else: 
				len_nd=len(Ixs_c[-1*nd+trj_cnt-1])
			
			n+=len_nd		#number of particles for class c


		P+=[(n+0.0)/N]	#add the probability of the new class to P

		#generate n particles of class c
		P_c+=[c]*n

		Ixs_c+=[range(y,y+n)]

		y=y+n
	

	#######################################################################

	P_w=[1.0/N]*len(P_c)

	##########################################################################

	return P_c,P_w,P 	#particle class, weight, class probability
		
#######################################################################


def mpf(lbl,coarse_ratio,coarse_and_fine,trj_cnt,observation_noise,abrupt,once,once_gamma,abrupt_ratio,write_files,print_out, dynamics_noise=0.01, convergence_time_mode=0, dist_threshold=0.1,abrupt_once=0):

	'''

	THE MAIN SCRIPT STARTS HERE

	'''
	converged=0

	#pdb.set_trace()

	'''
	lbl=None
	if len(sys.argv)>1:
		lbl=int(float(sys.argv[1]))
	
	coarse_ratio=0
	if len(sys.argv)>2:
		coarse_ratio=float(sys.argv[2])

	coarse_and_fine=0
	if len(sys.argv)>3:
		coarse_and_fine=float(sys.argv[3])

	trj_cnt=0
	if len(sys.argv)>4 and sys.argv[4]!="_":
		trj_cnt=int(float(sys.argv[4]))

	once=0  #wether the coarse observation will happen only for a small window once
	if len(sys.argv)>7 and sys.argv[7]!="_":
		once=int(float(sys.argv[7]))

	once_gamma=0  	#level at which the observation applies
	if len(sys.argv)>8 and sys.argv[8]!="_": 
		once_gamma=int(float(sys.argv[8]))

	'''

	traj_marker=-1  #symbol that separates the trajectories in data file
	frechet_clust=1


	if lbl is None:
		lbl_s=''
	else:
		lbl_s=str(lbl)


	'''
	abrupt=0
	if len(sys.argv)>6 and sys.argv[6]!="_":
		abrupt=float(sys.argv[6])   #if abrupt is one, the observations will be fine for coarse_ratio percent of time then become coarse only for coarse_ration percent, and so on...


	abrupt_ratio=0
	if len(sys.argv)>9 and sys.argv[9]!="_":
		abrupt_ratio=float(sys.argv[9])

	write_files=1
	if len(sys.argv)>10 and sys.argv[10]!="_":
		write_files=float(sys.argv[10])

	'''
	######################################################
	#if abrupt>0:
	#	abrupt_ratio=coarse_ratio
	#	coarse_ratio=0
	######################################################



	'''
	label='mouse_33333'+'___'+lbl_s
	if abrupt==1:
		label+='+'
	data_file='mouse_33_.dat'
	dist_file='mouse_33_dists.dat'
	test_file='mouse_33_test_'+lbl_s+'.dat'

	x_min=0
	x_max=1
	y_min=0
	y_max=1



	label='Bejing_175___'+lbl_s
	data_file='Bejing_175.dat'
	dist_file='dists_Bejing_175.dat'
	test_file='Bejing_full_test_'+lbl_s+'.dat'

	x_max=40.211
	x_min=39.7622
	y_min=116.0715
	y_max=116.8023





	label='walking_'+lbl_s
	data_file='walking.dat'
	dist_file='dists_walking.dat'
	test_file='walking_test_'+lbl_s+'.dat'

	x_max=222.97
	x_min=192.78

	y_max=363.30
	y_min=347.78
	'''

	'''
	label='Uber_'+lbl_s
	data_file='Uber_500_2.dat'
	dist_file='Uber_dists_500_2.dat'
	test_file='Uber_test_'+lbl_s+'.dat'
	'''
	
	
	label='Uber_centred_'+lbl_s
	data_file='Uber_centred.dat'
	dist_file='Uber_centred_dists.dat'
	test_file='Uber_centred_test_'+lbl_s+'.dat'
	

	'''
	label='mouse_13_'+lbl_s
	data_file='mouse_13.dat'
	dist_file='mouse_13_dists.dat'
	test_file='mouse_13_test_'+lbl_s+'.dat'

	'''

	D=numpy.loadtxt(data_file) #load data from file

	ends=[-1]+([x for x in range(len(D)) if D[x,0]==traj_marker and D[x,1]==traj_marker ])	#indices of end of trajectories


	x_max=max(D[:,0])
	if x_max==-1: x_max=max([m if i not in ends else -10000 for i,m in enumerate(D[:,0])])

	x_min=min(D[:,0])
	if x_min==-1: x_min=min([m if i not in ends else 10000 for i,m in enumerate(D[:,0])])

	y_max=max(D[:,1])
	if y_max==-1: y_max=max([m if i not in ends else -10000 for i,m in enumerate(D[:,1])])

	y_min=min(D[:,1])
	if y_min==-1: y_min=min([m if i not in ends else 10000 for i,m in enumerate(D[:,1])])


	#print x_min,x_max
	#print y_min, y_max


	rng=math.sqrt(abs(x_max-x_min)*abs(x_max-x_min)+abs(y_max-y_min)*abs(y_max-y_min)) #maximum range of distances

	observation_noise=observation_noise*rng
	dynamics_noise=dynamics_noise*rng 
	'''
	
	if len(sys.argv)>5 and sys.argv[5]!="_":
		observation_noise=float(sys.argv[5])
	'''

	if trj_cnt==0:
		trj_cnt=len(ends)-1 			#number of trajectories		
	print "Number of trajectories ",trj_cnt

	Trajs=dict()	
	for i in range(len(ends)-1):
		Trajs[i]=D[ends[i]+1:ends[i+1]]
	

	gamma=0

	if os.path.isfile(test_file): 	#check test file exists
		test=numpy.loadtxt(test_file)
		lbl=None
	else:			#else use one of the trajectories -- cross validation
		test=Trajs[lbl]


	##############################################################################################
	##############################################################################################
	#lbl=None
	##############################################################################################
	##############################################################################################

	#pdb.set_trace()

	print trj_cnt

	#using Frechet distance
	if frechet_clust:	
		if os.path.isfile(dist_file):	#can load distances from a file?
			dists=numpy.loadtxt(dist_file)

			if dists.shape[0]!=trj_cnt:
				dists.shape=(trj_cnt,trj_cnt)	#change the shape to square
		else:		#recompute distances
			dists=[[[] for i in range(trj_cnt)] for j in range(trj_cnt)]

			for i in range(trj_cnt):
			    for j in range(trj_cnt):
				dists[i][j]=round(frechet.frechetDist(Trajs[i],Trajs[j]),3)

			numpy.savetxt(dist_file,dists,'%.2f')


	
		[tree, C, parent]=cluster_frechet_hierarchical.cluster_frechet(dist_file,-1)
			
	else:
		# a different way to generate the requird data structures
		[tree, C, parent]


	####################################


	
	parent[-len(tree)]=-len(tree)



	done=[]
	ctree=[]
	clusts=[]

	for i,t in enumerate(tree):	
		j=-i-1
		clust=[]

		if j in done:
			continue
		while j!=len(tree)-1:
			if j==-len(tree) or tree[-parent[j]-1].distance-t.distance>threshold: break
			clust.append(j)
		
			clust.append(parent[j])
			if j!=-len(tree): j=parent[j]

		if len(clust)==0:
			ctree.append(CTreeNode(tree[i].distance,[tree[i].right,tree[i].left],-i-1,parent[-i-1] if -i-1!=-12 else -100))
		else:
			done+=clust
			ctree.append(CTreeNode(max([tree[nd].distance for nd in clust]),
				list(itertools.chain.from_iterable([[tree[-nd-1].right for nd in list(set(clust)) if tree[-nd-1].right not in clust] ,  [tree[-nd-1].left  for nd in list(set(clust)) if tree[-nd-1].left  not in clust]  ]))  , min(clust) , parent[min(clust)] ))#C[-1-nd]

		clusts.append(clust)
		
	for c in sorted(clusts,reverse=False):
		if len(c)!=0:
			for node in c:
				for key,value in parent.items():				
					if parent[key]==node:
						parent[key]=min(c)
					
					
	tree=ctree	
	


	####################################



	

	birth=[0]*trj_cnt+[x+1 for x in range(len(tree))]	#birth index

	birth_d=[0]*trj_cnt+[tree[l].distance for l in range(len(tree))] 	#birth distance

	b_d=numpy.array(birth_d)
	i=0
	maxi=max(b_d)+1000;


	while(not all(b_d==maxi)):
		min_val=min(b_d)
		min_indices=numpy.where(b_d==min_val)
		#define a layer of these 
		birth=[i if j in min_indices[0] else x for j,x in enumerate(birth)]
	
		#birth[min_indices]=i
		i=i+1
		#birth_d[min_indices]=min(birth_d)
		birth_d=[min_val if j in min_indices[0] else x for j,x in enumerate(birth_d)]
	
		#b_d[b_d==min_val]=maxi
		b_d=[maxi if j in min_indices[0] else x for j,x in enumerate(b_d)]




	cls_cnt=trj_cnt+len(tree) 	#number of classes/tree nodes
	
	prev_dist=-1;
	id=-1;
	I=range(cls_cnt-1) 	#all paths and clusters but root
	death_d=range(cls_cnt-1) 
	for n in tree: #id  will be 0 for -1, 1 for -2,... -1*id-1 to get the cluster number
	    if n.distance!=prev_dist:
		id=id+1

	    for nd in n.nodes:
		if nd>=0:	#normal trajectory
			if I[nd]==nd:	#not changed yet
				I[nd]=id+1
				death_d[nd]=n.distance
		else:		#a cluster already
			if I[-1*nd+trj_cnt-1]==-1*nd+trj_cnt-1:	#not changed yet
				I[-1*nd+trj_cnt-1]=id+1
				death_d[-1*nd+trj_cnt-1]=n.distance


	   
	    
	    prev_dist=n.distance	

	death=I+[max(I)+1]
	death_d=death_d+[float("inf")]

	#death_d=[tree[l-1].distance for l in I]+[tree[len(tree)-1].distance]	#death_distance



	'''
	print 'tree',tree
	print 'birth', birth
	print 'birth_d', birth_d
	print 'death', death
	print 'death_d', death_d
	print 'count',trj_cnt
	print 'tree length',len(tree)

	'''
	### initialise ########################

	N=100 # number of particles in the finset level

	P_pos=[[0,0]]*N 	# initial positions
	P_c=[0]*N 	# particle classes

	P=[0.0]*(cls_cnt)	#class probabiliies

	#define a uniform prior
	trajs_Ix=range(0,trj_cnt)
	if lbl is not None:
		trajs_Ix.remove(lbl)

	for i in range(N):	#for every particle
		if len(trajs_Ix)==0:	
			trajs_Ix=range(0,trj_cnt)
			if lbl is not None:
				trajs_Ix.remove(lbl)
		r1=random.randint(0,len(trajs_Ix)-1)
		r3=trajs_Ix[r1]	#index of trajectory, chosen randomly without replacement from the finest level
		trajs_Ix.remove(r3)

		P_c[i]=r3
	
		P_pos[i]=Trajs[r3][0,:]		#initial position of chosen trajectory
	
	
	
	Ixs_c=[[] for i in range(trj_cnt)]	 #indices of classes in particle set
	for c in range(trj_cnt):
		Ixs_c[c]=numpy.where(numpy.array(P_c)==c)



	P_c,P_w,P=rebuild_tree(trj_cnt,P_c,tree,N)	#generate the rest of the tree from the finest level
	#assert abs(sum([P_w[y] for y in range(N)])-1)<0.01


	#find initial positions for new particles in the tree
	for i in range(N,len(P_c)):	#all the particles other than the N ones in the finest level
		c=P_c[i] #get class of particle
		#get traj ids of class
		traj_ids=C[c-trj_cnt]		# multiple trajectory ids
		r1=random.randint(0,len(traj_ids)-1)	
		P_pos+=[Trajs[traj_ids[r1]][0,:]]
	

	prtcl_cnt=len(P_c)		# the total number of particles
	P_w=[1.0/N]*prtcl_cnt		# uniform weight for all particles, sums to 1 at every level


	#done, we have the tree with N*(1+len(tree)) particles, with all probability distributions at all levels proper





	T=len(test)

	X=[]
	Y=[]
	W=[] #histogram data structures
	Pr=[]
	X0=[]

	entropy=[]

	map_dists=[]
	exp_dists=[]
	P_once_gamma=[]

	map_dist_tot=0.0
	map_dist_win5=[]

	kdds=dict()

	#############################################################
	#if not os.path.exists('../gpsdata/class_figs/'+ label):
	#    os.makedirs('../gpsdata/class_figs/'+ label)
	#fig=plt.figure(1)
	#############################################################

	#create kdds for all classes
	for c in range(cls_cnt):
		if c<trj_cnt:
			DD=Trajs[c]
			#############################################################
			'''			
			fig.clf()
			plt.plot(Trajs[c][0:len(Trajs[c])-1,0],Trajs[c][0:len(Trajs[c])-1,1],linewidth=2.5)
			plt.axis((x_min,x_max,y_min,y_max))
			plt.savefig('../gpsdata/class_figs/'+ label +'/label_'+str(c)+'.png', bbox_inches='tight')
			'''
			#############################################################
		else:
			DD=[-1,-1] 	#############################################################
		
			for h in C[c-trj_cnt]: # for each of the trajectories in the class
				d1=Trajs[h][0:Trajs[h].shape[0]]
					
				DD=numpy.vstack([DD,d1[:,[0,1]]]) # collate all the examples from all trajectories
	
		kdds[c]=sp.cKDTree(DD)


	Obs=[0]*T


	#pdb.set_trace()


	#-----------------------------------------------------
	#The main loop
	for t in range(T):
		if print_out: print 'step', t

		#an array of indices of particles in every class
		Ixs=[0]*(len(tree)+trj_cnt)

		for c in range(len(tree)+trj_cnt):
			I=numpy.where(numpy.array(P_c)==c)
			I = [map(int, x) for x in I][0]
			Ixs[c]=I


		norm=0.0 	#normalisation term
		Is=[]	#indices of updated particles
	
		#assert len(P_c)==len(P_w)
		#assert len(P_pos)==len(P_w)


		#get new noisy observation from test trajectory
		
		o=numpy.array(test[t])+numpy.array([observation_noise*random.random(),observation_noise*random.random()])
		o=o.tolist()
	
		####################################################################
		#print tree, parent
		if once>=1 and once<=3 and coarse_ratio>0 and t>coarse_ratio*T:	#update once depending on coarse_ratio
			once=once+1
			coarse_obs=1
			if once_gamma==0:
				o_c=most_probable_class(o,observation_noise, cls_cnt, kdds,rng,lbl,trj_cnt,C) 	# observed class
				gamma=(birth[o_c]+death[o_c])/2.0
			else:
				if lbl is not None:
				#gamma is specified as input, find o_c from gamma for the test trajectory
				#trace the parent up until gamma
					gamma=once_gamma
					p1=lbl
					while True:
						if print_out: print p1, birth[trj_cnt-p1-1], death[trj_cnt-p1-1]
						if birth[trj_cnt-p1-1]<=gamma and death[trj_cnt-p1-1] > gamma:
							break
						p1=parent[p1]
					
					#p1 is the class
					o_c=trj_cnt-p1-1

			
			
			if print_out: print 'observation', o_c, gamma
			Obs[t]=o_c
		elif once<1 and random.random()>1-coarse_ratio: 	# a coarse observation
			coarse_obs=1
			o_c=most_probable_class(o,observation_noise, cls_cnt, kdds,rng,lbl,trj_cnt,C) 	# observed class
			gamma=(birth_d[o_c]+death_d[o_c])/2.0
			if print_out: print 'observation', o_c, gamma
			Obs[t]=o_c
		else:
			coarse_obs=0
			gamma=0
			if print_out: print 'observation', o, gamma
	
			Obs[t]=-100
		#####################################################################
				
		P_w2=[0]*len(P_w)
		#pdb.set_trace()
		#------------------------------------------------------------
		#do for every particle
		for i in range(prtcl_cnt):

			#-----------------------------------------------------------------------------	
			#update positions
			#-----------------------------------------------------------------------------

			#project the particle forward according to class model
			try:
				c=P_c[i] # get class
			except:
				if print_out: print i, len(P_c)
				exit() 
			z=P_pos[i] # get position

			if c<trj_cnt: 	#a single trajectory class
				d=Trajs[c] # data points of class
			
				#prepare dynamics data matrices [x,y,vx], [x,y,vy]
				d1=d[0:d.shape[0]-1]
				d2=d[1:d.shape[0]]
				d3=d2-d1
				Dx=numpy.append(d1,d3,1)
				Dx=Dx[:,[0,1,2]]
				Dy=numpy.append(d1,d3,1)
				Dy=Dy[:,[0,1,3]]

				cps=1 	# number of components (trajectories)
				#find the maximum distance between points to get the path as epsilon
				e=max(numpy.sum(numpy.diff(d,1,0),1))
			
		
			else:		#multiple trajectories class
				e=tree[c-trj_cnt].distance	# epsilon value depends on the filtration value
			
				#prepare dynamics data matrices [x,y,vx], [x,y,vy]
				d=[-1,-1]
				Dx=[-1,-1,-1]
				Dy=[-1,-1,-1]
				cps=len(C[c-trj_cnt])	# number of components (trajectories)

				for h in C[c-trj_cnt]: # for each of the trajectories in the class
					dd=Trajs[h] # data points of class
					d1=dd[0:dd.shape[0]-1]
					d2=dd[1:dd.shape[0]]
					d3=d2-d1
					d13=numpy.append(d1,d3,1)
				
					Dx=numpy.vstack([Dx,d13[:,[0,1,2]]]) # collate all the examples from all trajectories
					Dy=numpy.vstack([Dy,d13[:,[0,1,3]]])
					d=numpy.vstack([d,dd])

				

		
				#remove the first extra item
				d=numpy.delete(d,0,0)
				Dx=numpy.delete(Dx,0,0)
				Dy=numpy.delete(Dy,0,0)

			

		
			# get indices of nearby poitns 
			kdx=sp.cKDTree(Dx[:,[0,1]])
		
			dix=[]
			factor=16	#find enough points to compute dynamics, try smaller balls first
			while(len(dix)<20):
				dix=kdx.query_ball_point(z,e/factor)	#find all points in Dx which are e away from current z
				if factor==1:
					break
				factor=factor/2
		


			#theta1=float('nan')
			#theta2=float('nan')
		
			if len(dix)==1: # a single point -- copy velocity
				vel_x=Dx[dix,2][0]
				vel_y=Dy[dix,2][0]
			
			elif dix!=[] and len(dix)!=1: 	# multiple points  -- fit a model
				if len(dix)>20:
					dix=random.sample(dix, 20)
				vel_x=dynamics_lin2(Dx[dix,:],z,rng, dynamics_noise)
				vel_y=dynamics_lin2(Dy[dix,:],z,rng, dynamics_noise)
			
			

			elif dix==[]:
				dix=kdx.query_ball_point(z,1.5*e)	#find all points in Dx which are 2*e away from current z
				vel_x=dynamics_lin2(Dx[dix,:],z,rng, dynamics_noise)
				vel_y=dynamics_lin2(Dy[dix,:],z,rng, dynamics_noise)

			
			# update position
			P_pos[i]=[z[0]+vel_x, z[1]+vel_y]	

			#-----------------------------------------------------------------------------	
			#update weights
			#-----------------------------------------------------------------------------
			#update particle only if in the right class
			if gamma>=birth_d[c]  and gamma<death_d[c]-0.0000001:	#strange behaviour (2<2.0)
				#print i,birth[c],gamma,death[c]
				p_old=P_w[i]
				p=-1
			
				if (not coarse_obs) or (coarse_and_fine):
					#if i==0 and print_out: print 'update fine'
					p=(update_particle(o,z,c,rng))
					#if print_out: print p


				#for a rough class o (number of class)
				#using the class observation, compute distance for each particle to the new class observed,

				if coarse_obs and abrupt!=2:	#abrupt, coarse phase, fine filter, don't update
					if p==-1: p=update_particle_c(o_c,P_c[i],trj_cnt,parent,rng, tree)
					else: p+=update_particle_c(o_c,P_c[i],trj_cnt,parent,rng, tree)

					#if i==0 and print_out: print 'update coarse'
					####################################################				
					#p+=update_particle_c(o_c,P_c[i],trj_cnt,parent,rng)
					####################################################
					
				if p!=-1:
					Is=Is+[i] 	#keep a sum of all weights and all updated indices
					norm=norm+p
					P_w2[i]=p
				else:
					print 'NOT UPDATED'
					
		
		#all particles done ###
		#-----------------------------------------------------------------------------	
		#Housekeeping
		#-----------------------------------------------------------------------------				
		#if print_out: pdb.set_trace()
		if len(Is)!=0 and sum([P_w2[x] for x in Is])!=0: 	#possibly no particles exist from this class/gamma?
			#normalise all updated particles
			if norm!=0:
				for i in Is: 	#updated particle indices
					P_w2[i]/=(norm)#+0.00000000001)
			if abs(sum([P_w2[x] for x in Is])-1)>=0.15 and print_out:
				pdb.set_trace()
	
			#assert abs(sum([P_w2[x] for x in Is])-1)<0.15


			#if print_out: print P_w
			#rebalance the tree after update 
	
			'''alive classes:', [P_c[x] for x in range(prtcl_cnt) if birth[P_c[x]]<=gamma and  death[P_c[x]] > gamma]'''
			
			if abs(sum([P_w[x] for x in range(prtcl_cnt) if birth_d[P_c[x]]<=gamma and  death_d[P_c[x]]-0.0000001 > gamma])-1)>=0.15:
				pdb.set_trace()


			######################################################
			for x in Is:
				P_w[x]=0.75*P_w[x]+0.25*P_w2[x] 	
			######################################################
			if abs(sum([P_w[x] for x in range(prtcl_cnt) if birth_d[P_c[x]]<=gamma and  death_d[P_c[x]]-0.0000001 > gamma])-1)>=0.15:
				pdb.set_trace()
			#assert abs(sum([P_w[x] for x in range(prtcl_cnt) if birth_d[P_c[x]]<=gamma and  death_d[P_c[x]]-0.0000001 > gamma])-1)<0.15

		#all classes that should have been updated if they have particles
		updated= [x for x in range(cls_cnt) if birth_d[x]<=gamma and death_d[x]-0.0000001>gamma]

		for x in updated:
			P[x]=0	

		# compute current class probabilities
		#lower level first
		for i in range(prtcl_cnt):
			if P_c[i] in updated:
				P[P_c[i]]+=P_w[i]

		#print P

		#print updated
		#print sum([P[x] for x in updated])
		#assert abs(sum([P[x] for x in updated])-1)<0.15

	


		#update parents
		for j in sorted([j for j in range(len(tree)) if birth_d[trj_cnt+j]>gamma]):
			#update class trj_cnt+j by the sum of its children
			'''P[j]=sum([P[x] for x in C[j-trj_cnt]])'''

			#keep the old value
			old_P=P[trj_cnt+j]
	
			#index the distribution with appropriate index
			P[trj_cnt+j]=sum([P[nd] if nd>=0 else P[-1*nd+trj_cnt-1] for nd in tree[j].nodes])

			#compute the new weights of individual particles by ratio to old values
	
	
			if old_P!=0:
				for h in Ixs[trj_cnt+j]:
					P_w[h]=P_w[h]*P[trj_cnt+j]/old_P
			else:
				for h in Ixs[trj_cnt+j]:
					P_w[h]=P[trj_cnt+j]/len(Ixs[trj_cnt+j])



		#update children
		updated_parents=[]


		for j in [j for j in range(trj_cnt+len(tree)-2,-1,-1) if death_d[j]<=gamma]: 	#go through the tree, j=len(tree) --> j=1
			#find parent, should be updated by now


			if j<trj_cnt:
				prnt=-1*parent[j]-1+trj_cnt 
			else:
				prnt=-1*parent[-1*(j-trj_cnt)-1]-1+trj_cnt  	#get parent positive index

			if prnt in updated_parents:
				continue



			#update children of class prnt by the same percentage they had before the update
			children=[nd if nd>=0 else -1*nd+trj_cnt-1 for nd in tree[prnt-trj_cnt].nodes]




			
			sum_children=sum([P[x] for x in children])



			if sum_children!=0:		
				for c in children:
					old_P=P[c]			
					P[c]=(P[prnt]/sum_children)*P[c]
					#compute the new weights of individual particles
					for h in Ixs[c]:
						P_w[h]=P_w[h]*P[c]/old_P
					if abs(sum([P_w[x] for x in Ixs[c]])-P[c])>=0.1 and len(Ixs[c])!=0 and print_out:
						pdb.set_trace()
					#assert len(Ixs[c])==0 or abs(sum([P_w[x] for x in Ixs[c]])-P[c])<0.1
				#assert abs(sum([P[x] for x in children])-P[prnt])<0.1
			else:		#uniformly
				for c in children:
					old_P=P[c]
					P[c]=(1.0/len(children))*P[prnt]
					if old_P!=0:					
						#compute the new weights of individual particles
						for h in Ixs[c]:
							P_w[h]=P_w[h]*P[c]/old_P
						#assert abs(sum([P_w[x] for x in Ixs[c]])-P[c])<0.1
					else:
						for h in Ixs[c]:	#uniformly
							P_w[h]=P[c]/len(Ixs[c])
						#assert abs(sum([P_w[x] for x in Ixs[c]])-P[c])<0.1	
				#assert abs(sum([P[x] for x in children])-P[prnt])<0.1				
			updated_parents=updated_parents+[prnt]

		
		#assert abs(sum([P_w[y] for y in range(N)])-1)<0.1
		#-----------------------------------------------------------------------------	
		#Resampling
		#-----------------------------------------------------------------------------				
		#sample from the lowest level of the tree, then rebuild
		P_wheel=[0]*N 	# the first N particles
		#resample based on weights
		P_wheel[0]=P_w[0]
		for n in range(1,N):
			P_wheel[n]=P_wheel[n-1]+P_w[n]

		#assert abs(sum([P_w[y] for y in range(N)])-1)<0.1
		#assert abs(P_wheel[N-1]-1)<0.1

		P_c2=[0]*N

		#sample N particles based on weights
		for n in range(N):			
			I=[-1]

			while len(I)==0 or I[0]==-1:
				
				r=random.random()*sum([P_w[y] for y in range(N)])

				I=numpy.where(numpy.array(P_wheel)>r )
	
				I = [map(int, x) for x in I][0]
	

			P_c2[n]=P_c[I[0]]
		

		#randomly change 2% of particles to uniform distribution

		for n in range(int(N*1/100)):
			r=random.randint(0,N-1)
		
			new_class=random.randint(0,trj_cnt-1)
			if new_class!=lbl:
				P_c2[r]=new_class

		#pdb.set_trace()

		#rebuild the tree from low level particles

		P_c_old=P_c

		#assert abs(max(P)-1)<0.1
		P_c,P_w,P=rebuild_tree(trj_cnt,P_c2,tree,N)

		#assert abs(max(P)-1)<0.1

		#assert abs(sum([P_w[y] for y in range(N)])-1)<0.1

		#update P_pos to reflect the change
		P_pos2=[[0,0]]*len(P_c)
		for i,c in enumerate(P_c):
			#find all old ids where class was c
			I=numpy.where(numpy.array(P_c_old)==c)		
			I = [map(int, x) for x in I][0]
			if len(I)!=0: #previous places where this class was positioned
				r=random.randint(0,len(I)-1)
				P_pos2[i]=P_pos[I[r]]	
			else: 	#newly introduced class: choose a point based on parent position - if exists, else: random.
				if c<trj_cnt:
					prnt=-1*parent[c]+trj_cnt-1
				elif c==cls_cnt:	#root
					prnt=c
				else:
					prnt=-1*parent[-1*(c-trj_cnt+1)]+trj_cnt-1
				I=numpy.where(numpy.array(P_c_old)==prnt)	#previous places where parent class was positioned
				I = [map(int, x) for x in I][0]
				if len(I)!=0:	
					r=random.randint(0,len(I)-1)
					P_pos2[i]=P_pos[I[r]]	
				else:
					P_pos2[i]=P_pos[random.randint(0,len(P_pos)-1)]	
			


		P_pos=P_pos2

		prtcl_cnt=len(P_c)			
		#assert abs(sum([P_w[y] for y in range(N)])-1)<0.1

		'''
		coarse_ratio_=coarse_ratio
		coarse_ratio_=0.1
		if t%(1/coarse_ratio_)==0:
			#plot distribution of level at the single coarse obsevation
		
			g=195
			alive_P=[P[c] for c in range(cls_cnt) if birth[c]<=g and death[c] > g]
			print alive_P	
			#plt.plot(P)
		'''			

		classes_0=[x for x in range(len(birth)) if birth_d[x]<=0 and death_d[x]>0];P_0=[P[x] for x in classes_0];
		if print_out: print [x for x in classes_0 if P[x]==max(P_0)]

		x0=[x for x in classes_0 if P[x]==max(P_0)]
		X0+=x0
		map_dist_tot+=max([  round(class_dist1(x0_,lbl,parent, trj_cnt,tree),2) for x0_ in x0  ])
		
		if len(map_dist_win5)>=5: 
			del map_dist_win5[0] 	#remove the oldest element
		map_dist_win5+=[max([  round(class_dist1(x0_,lbl,parent, trj_cnt,tree),2) for x0_ in x0  ])]

		if convergence_time_mode and len(map_dist_win5)==5:
			if sum(map_dist_win5)/5.0<dist_threshold:
				converged=1
				break
		#-----------------------------------------------------------------------------	
		



		#deal with the abrupt observation case
		if abrupt>0:
			phase=round(T*abrupt_ratio)
			if t<phase:	#     ***F**%%%%%%%%%%COARSE%%%%%%%%**F***	
				if coarse_ratio==1 and print_out: print '----------- FINE -------------'
				coarse_ratio=0
			elif t>T-phase and abrupt_once==0:	#     ***F**%%%%%%%%%%COARSE%%%%%%%%COARSE	
				if coarse_ratio==1 and print_out: print '----------- FINE -------------'
				coarse_ratio=0
				
			else:
				if coarse_ratio==0 and print_out: print '---------- COARSE ------------'
				coarse_ratio=1	
				

		#save time-lapse snapshots of the process

		#if print_out: print P_c
		#pdb.set_trace()
	

		#compute average position  -- prediction
		#pos=numpy.sum(P_pos,0)/len(P_pos)
		if t%5==0:   #every 5 timesteps
			l=0
		
			#g=tree[l-1].distance

			#pos_l=[0,0]
			#nrm_l=0.0
			if once_gamma!=0:
			
				classes_once_gamma=[x for x in range(len(birth)) if birth[x]<=once_gamma and death[x]>once_gamma]	#alive classes at g
				#pdb.set_trace()
				if not P_once_gamma:
					P_once_gamma.append(', '.join([str(x) for x in classes_once_gamma]))
				
					#fig=plt.figure(1)
					for  h_,c_ in enumerate(classes_once_gamma):
						fig.clf()		
						img = imread("../gpsdata/class_figs/Uber_layer_499_0.png")
						#plt.imshow(img,zorder=0, extent=[x_min-0.025,x_max,y_min-0.008,y_max+0.008])

						#if c_<trj_cnt:		#plot the trajectory
								#plt.plot(Trajs[c_][1:len(Trajs[c_])-1,0],Trajs[c_][1:len(Trajs[c_])-1,1],linewidth=5, color=(0,0,0),zorder=1)
						#else:	#plot all the trajectories of the class
							#for i in C[c_-trj_cnt]:							
								#plt.plot(Trajs[i][1:len(Trajs[i])-1,0],Trajs[i][1:len(Trajs[i])-1,1],linewidth=5,color=(0,0,0),zorder=1)	
								#plt.plot(Trajs[i][1:2,0],Trajs[i][1:2,1],linewidth=10,color=(1,0,0),zorder=2, marker='x', markersize=10)	

						#plt.axis((x_min,x_max,y_min,y_max))
						#plt.axis('off')
						#plt.savefig('../gpsdata/class_figs/'+ label +'_layer_'+str(once_gamma)+'_'+str(h_)+'_'+str(c_)+'.png', bbox_inches='tight')




				P_once_gamma.append(', '.join(['%.3f' % P[x] for x in classes_once_gamma]))	#probabilities of classes at g
			
			#print classes_g
			#find particles from these classes
			#for i,c in enumerate(P_c):
			#	if c in classes_g:
			#		pos_l[0]+=P_w[i]*P_pos[i][0]
			#		pos_l[1]+=P_w[i]*P_pos[i][1]
			#		nrm_l+=P_w[i]
			#if(nrm_l)!=0:
			#	pos_l=numpy.array(pos_l)/nrm_l
	
			g=0
			classes_g=[x for x in range(len(birth)) if birth_d[x]<=g and death_d[x]>g] 	#alive classes at g
			if l==0 : 
				P_g=[P[x] for x in classes_g]	#probabilities of classes at g
				#if len(P_g)==0:
				#	continue
				#print P_g
				#assert sum(P_g)>1-0.1

				P_g_s=sorted(P_g, reverse=True)		#sorted class probabilities at g
				if len(P_g_s)!=1: P_g_s=[x+0.001 for x in P_g_s]
				P_g_i=sorted(range(len(P_g)), key=lambda k: P_g[k], reverse=True)  #indices of higher probability classes at g, top first

				for k in range(len(classes_g)):
					Pr+=[P[k]]
					X+=[classes_g[P_g_i[k]]] 	#most likely classes
					Y+=[t]
					W+=[P_g_s[k]]

				Pr+=[-100]
	
				entropy+=[round(sum([-x*math.log(x) for x in  P_g if x!=0]),2)]	#entropy of finest distribution
				#X0+=[classes_g[P_g_i[0]]]
			
				if lbl is not None:
					map_dists+=[round(class_dist1(P_g_i[0],lbl,parent, trj_cnt,tree),2)]#distance to MAP class at g
			
				sum_dists=0.0			
				for h in range(len(P_g_i) ):
					if lbl is not None:
						sum_dists+=P_g_s[h]*class_dist2(P_g_i[h],lbl,parent, trj_cnt+1,tree)


				exp_dists+=[round(sum_dists,2)]
	
	
		
		P_c_n=numpy.array(P_c)
		if print_out: print [sum(P_c_n==x) for x in range(cls_cnt) ]
		if print_out: print P
	#---------------------------------------------------------------------------------------------------------


	if write_files==1:
		#save the outputs to file

		'''f = open('obs.txt', 'w')
		for o in Obs:
		  f.write("%s\n" % o)

		f = open('probs_'+label+'_'+str(coarse_ratio)+'_'+str(coarse_and_fine)+'_'+str(once_gamma)+'+'+str(time.time())+'.csv', 'w')
		'''


		with open('probs_'+label+'_'+str(coarse_ratio)+'_'+str(coarse_and_fine)+'_'+str(once_gamma)+'_'+str(abrupt)+'.csv', "a") as f:
		    	f.write("%s\n" % map_dist_tot)



			#for p in Pr:
			#  f.write("%s\n" % p)
			#f.write("\n" )
			'''
			for p in X0:
			  f.write("%s\n" % p)
			f.write("\n" )



			for p in entropy:
			  f.write("%s\n" % p)
			f.write("\n" )
			for p in map_dists:
			  f.write("%s\n" % p)
			f.write("\n" )




			for p in exp_dists:
			  f.write("%s\n" % p)
			'''
			#if once_gamma!=0:
			#	for p in P_once_gamma:
			#  		f.write("%s\n" % p)
		f.close()


	if convergence_time_mode==1: 
		if converged==1: return float(t)/T
		else: return -1
	else: 
		return map_dist_tot/T
