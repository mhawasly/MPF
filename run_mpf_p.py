from dat_to_voronoi_3__keep_lbl_in_gamma_d import mpf 
import itertools
from multiprocessing import Pool, freeze_support
import os.path


num_tests=10

num_repeats=25

ratios=[0.5,0.33]#,0.2,0.1,0.05,0]

dynamics_noise=[0.1,0.3,0.5,0.75]#0.01,0.05,,0.2

observation_noise=[0.01,0.05,0.1] #coarseness



def single_test(i,x,r,n,c):
	res=0
	res2=0
	for j in range(num_repeats):
		#res+=mpf(i,0,0,0,c,x,0,0,r,0,0,n)	 #abrupt (x=1 => fine-coarse-fine, x=2 => fine-...-fine)
		if not os.path.isfile('mpf_res_'+str(i)+'_cf_'+str(r)+'_'+str(n)+'_'+str(c)+'.csv'):
			res+=mpf(i,r,1,0,c,0,0,0,0,0,0,n)	# coarse & fine
		if r==ratios[0] and not os.path.isfile('mpf_res_'+str(i)+'_f_0_'+str(n)+'_'+str(c)+'.csv'): res2+=mpf(i,0,0,0,c,0,0,0,0,0,0,n)	# fine
	
	
	'''with open('mpf_res_'+str(i)+'_'+str(x)+'_'+str(r)+'_'+str(n)+'_'+str(c)+'.csv', "a") as f:
		f.write("%s" % res)
		f.close()
	'''

	if not os.path.isfile('mpf_res_'+str(i)+'_cf_'+str(r)+'_'+str(n)+'_'+str(c)+'.csv'):
		res/=num_repeats
		with open('mpf_res_'+str(i)+'_cf_'+str(r)+'_'+str(n)+'_'+str(c)+'.csv', "a") as f:
			f.write("%s" % res)
			f.close()

	if r==ratios[0] and not os.path.isfile('mpf_res_'+str(i)+'_f_0_'+str(n)+'_'+str(c)+'.csv'):
		res2/=num_repeats
		with open('mpf_res_'+str(i)+'_f_0_'+str(n)+'_'+str(c)+'.csv', "a") as f:
			f.write("%s" % res2)
			f.close()


def single_test_star(args):
	return single_test(*args)



def main():
	p = Pool(num_tests)

	args_i = range(num_tests)
	args_x=[1]#,2]
	args_r=ratios
	args_n=dynamics_noise
	args_c=observation_noise


	iterables = [ args_i, args_x, args_r , args_n, args_c]

	Args=[]
	for t in itertools.product(*iterables):
	    Args.append(t)
	
	print Args

	p.map(single_test_star, Args)


if __name__=="__main__":
    freeze_support()
    main()
	

