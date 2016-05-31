import math

import numpy as np
import matplotlib.mlab as mlab
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

num_tests=10


ratios=[0.5,0.33]#[0.5,0.33,0.2,0.1,0.05,0]

noise=[0.1,0.3,0.5,0.75]#[0.01,0.05,0.1,0.2,0.3,0.5]

coarseness=[0.1,0.05,0.01] #[0.01,0.05,0.1]

args_i = range(num_tests)
args_x=[1];#,2]
args_r=ratios





	
for j,r in enumerate(ratios):
	f, axarr = plt.subplots(len(coarseness),len(noise))
	for h,n in enumerate(noise):
		for u,c in enumerate(coarseness):
			for x in args_x:
				Out=[]
				Vals=[]
				
				Out_cf=[]
				Vals_cf=[]
				
				Out_f=[]
				Vals_f=[]
				b=0

				for i in args_i:
					try:
						'''with open('mpf_res_'+str(i)+'_'+str(x)+'_'+str(r)+'_'+str(n)+'_'+str(c)+'.csv', "r") as f:
							val=float(f.read())
							f.close()
						'''
						with open('mpf_res_'+str(i)+'_cf_'+str(r)+'_'+str(n)+'_'+str(c)+'.csv', "r") as f:
							val_cf=float(f.read())
							f.close()
						with open('mpf_res_'+str(i)+'_f_0_'+str(n)+'_'+str(c)+'.csv', "r") as f:
							val_f=float(f.read())
							f.close()
						
					except EnvironmentError:
						b=1
						break
				


					#Vals.append(val)
					#Out.append(str(i)+'\t'+str(val))

					Vals_cf.append(val_cf)
					Out_cf.append(str(i)+'\t'+str(val_cf))
					Vals_f.append(val_f)
					Out_f.append(str(i)+'\t'+str(val_f))
				
				if b==1: break
	
				#avg=float(sum(Vals))/len(Vals)
				#std=math.sqrt(float(sum([(v-avg)**2 for v in Vals]))/len(Vals))

				avg_cf=float(sum(Vals_cf))/len(Vals_cf)
				std_cf=math.sqrt(float(sum([(v-avg_cf)**2 for v in Vals_cf]))/len(Vals_cf))

				avg_f=float(sum(Vals_f))/len(Vals_f)
				std_f=math.sqrt(float(sum([(v-avg_f)**2 for v in Vals_f]))/len(Vals_f))

				#with open('totals/mpf_res_totals_'+'_n'+str(n)+'_r'+str(r)+'_c'+str(c)+'_'+str(x)+'.csv', "w") as f:


				
				x = np.linspace(-0.5,1.5,100)
				axarr[u, h].plot(x,mlab.normpdf(x,avg_cf,std_cf),c='red')
				for v in Vals_cf:
					axarr[u, h].plot([v,v],[0,1],c='red')


				axarr[u, h].plot(x,mlab.normpdf(x,avg_f,std_f),c='blue')
				for v in Vals_f:
					axarr[u, h].plot([v,v],[0,1],c='blue')

				if u==len(coarseness)-1: axarr[u, h].set_xlabel(str(n)+' dyn' )
				if h==0: axarr[u, h].set_ylabel(str(c)+' obs' )

				axarr[u, h].set_ylim([0,10])

				#plt.close()

				with open('totals/mpf_res_totals_'+'_n'+str(n)+'_r'+str(r)+'_c'+str(c)+'_'+'.csv', "w") as f:
					#for o in Out:
						#f.write(o+'\n')
					#f.write(str(avg)+'\n')
					#f.write(str(std)+'\n')
					#f.close()

					for o in Out_cf:
						f.write(o+'\n')
					f.write(str(avg_cf)+'\n')
					f.write(str(std_cf)+'\n')

					f.write('==========\n')

					for o in Out_f:
						f.write(o+'\n')
					f.write(str(avg_f)+'\n')
					f.write(str(std_f)+'\n')
					f.close()


	plt.setp(axarr,  yticks=[])
	plt.setp(axarr,  xticks=[0,0.75,1.5])
	#plt.xlabel('Observation noise')
	#plt.ylabel('Coarseness period')
	plt.suptitle(str(int(r*100))+'% coarse')
	plt.subplots_adjust(top=0.85,wspace=0.4,hspace=0.4)#,right=0.85, left=0.84)

	#plt.tight_layout(pad=1.4, w_pad=0.5, h_pad=2.0)
	plt.savefig('figs/mpf_res_totals_'+'_r'+str(r)+'.png') #+'_r'+str(r)+'_c'+str(c)+'__
	plt.close()
	

