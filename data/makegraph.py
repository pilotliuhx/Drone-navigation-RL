import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

filename = ["world0/log0.txt","world1/log1.txt","world2/log2.txt","world3/log3.txt","world4/log4.txt","world5/log5.txt",]
# extract out x's and y's
i=0
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 3)
ylim = []
for fn in filename:
    with open(fn, 'r') as f:
        x, y, avey, uper, lower, upert,lowert= [], [], [], [], [],[],[]
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Average Episodic Return' in l:
                y.append(float(l[1]))
            if 'Timesteps So Far' in l:
                x.append(int(l[1]))
        if i == 3:
            lower.append(y[0]-1)
            lowert.append(x[0]-200)
        for tmp in range( len(y)):
            ave = 0
            if tmp<len(y)-10:
                for n in range(10):
                    ave = ave + y[tmp+n]
                ave = ave / 10
                avey.append(ave)
                if y[tmp] > ave:
                    uper.append(y[tmp])
                    upert.append(x[tmp])
                else:
                    lower.append(y[tmp])
                    lowert.append(x[tmp])
        if i == 3 or i==4:
            lower.append(y[tmp]-1)
            lowert.append(x[tmp])
    ax = fig.add_subplot(gs[i])
    #ax.plot(x, y, alpha=0.5,linewidth=1,linestyle='-',label='per episode')
    ax.fill_between(upert,uper,-50,color='g', alpha=0.3)
    ax.fill_between(lowert,lower,-50,fc='white',alpha=1)
    ax.plot(x[0:-10], avey,color='g', alpha=1,linewidth=1.2,linestyle='-',label='average of 10 episodes')
    #ax.legend(loc='lower right',fontsize=12)
    # if i == 0:
    #     ylim=ax.get_ylim()
    # else:
    #     ax.set_ylim(ylim)
    ax.set_ylim(-45,30)
    if i != 3 and i !=5:
        ax.set_xlim(0,3e6)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    ax.set_title('Environment '+str(i),size=20,family='Times New Roman',fontweight='bold')
    ax.set_xlabel('trainning steps',size=18,family='Times New Roman')
    ax.set_ylabel('return per episode',size=18,family='Times New Roman')
    i=i+1
# plt.legend(['env0', 'env1', 'env2', 'env3', 'env4', 'env5'])
# plt.xlabel('Average Total Timesteps So Far')
# plt.ylabel('Average Episodic Return')
# Show graph so user can screenshot
plt.show()