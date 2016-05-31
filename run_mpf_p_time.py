from dat_to_voronoi_3__keep_lbl_in import mpf 
import itertools
from multiprocessing import Pool, freeze_support
import numpy

num_tests=10

num_repeats=25

ratios=[0.01,0.05,0.075,0.1,0.15,0.2,0.25]#,0.1,0.05,0]'''0.75,'''0.5,0.33,

dynamics_noise=[0.75]#[0.3,0.5]#,0.66,0.75]#0.2,[0.01,0.05,0.1

observation_noise=[0.01]#,0.05,0.1] #coarseness

thresholds=[0.33]#0.1,,0.5

def single_test(i,x,r,n,c,t):
	res=[]
	res2=[]
	res2_avg=-1
	res2_std=-1
	res_avg=-1
	res_std=-1

	#abrupt with ratio r
	for j in range(num_repeats):
		try:
			r1=mpf(i,0,0,0,c,1,0,0,r,0,0,n,1,t,1)	#abrupt_once set
			if r1!=-1: res+=[r1]	# coarse & fine
		except Exception:
			print 'Exception!',[i,0,0,0,c,1,0,0,r,0,0,n,1,t]
		try:
			r2=mpf(i,0,0,0,c,2,0,0,r,0,0,n,1,t,1)	#abrupt_once set
			if r2!=-1: res2+=[r2]	# fine
		except Exception:
			print 'Exception!',[i,0,0,0,c,1,0,0,r,0,0,n,1,t]

	'''#coarse rate
	for j in range(num_repeats):
		r1=mpf(i,r,1,0,c,0,0,0,0,0,0,n,1,t)
		if r1!=-1: res+=[r1]	# coarse & fine
		if r==ratios[0]: 
			r2=mpf(i,0,0,0,c,0,0,0,0,0,0,n,1,t)
			if r2!=-1: res2+=[r2]	# fine
	'''


	if len(res)!=0:
		res_avg=sum(res)/len(res)
		res_std=numpy.std(res)
	
	'''r==ratios[0] and'''
	if  len(res2)!=0:
		res2_avg=sum(res2)/len(res2)
		res2_std=numpy.std(res2)

	with open('mpf_res_TH'+str(t)+'_'+str(i)+'_cf_'+str(r)+'_'+str(n)+'_'+str(c)+'_abrupt.csv', "w") as f:
		for rr in res:
			f.write("%s\n" % rr)
		f.write("------\n")
		f.write("%s\n" % res_avg)
		f.write("%s\n" % res_std)
		f.close()

	'''if r==ratios[0]:'''

	with open('mpf_res__TH'+str(t)+'_'+str(i)+'_f_'+str(r)+'_'+str(n)+'_'+str(c)+'_abrupt.csv', "w") as f:
		for rr in res2:
			f.write("%s\n" % rr)
		f.write("------\n")
		f.write("%s\n" % res2_avg)
		f.write("%s\n" % res2_std)


def single_test_star(args):
	return single_test(*args)



def main():
	p = Pool(num_tests)

	args_i = range(num_tests)
	args_x=[1]#,2]
	args_r=ratios
	args_n=dynamics_noise
	args_c=observation_noise
	args_t=thresholds


	iterables = [ args_i, args_x, args_r , args_n, args_c, args_t]

	Args=[]
	for t in itertools.product(*iterables):
	    Args.append(t)
	
	#print Args

	p.map(single_test_star, Args)


if __name__=="__main__":
    freeze_support()
    main()
	

