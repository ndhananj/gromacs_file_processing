################################################################################
# Do the steps needed to create a pmf
# As a module will import the function that generates data as well
# original author: Nithin Dhananjayan (ndhananj@ucdavis.edu)
# Usage : python <this_file> <prefix> <num_frames> <spring_constant>
# Example : python pmf_calcs.py timeStep 10 250
################################################################################

from gmx_file_processing import *

out_prefix = 'timeStep'

def gen_pmf_data(orig_trj,struct_file, beg, end, step,\
    mdp_file, top_file, ndx_file, ndx):
    get_frames_from_trj(orig_trj, struct_file, beg, end, step, out_prefix)
    numFrames=np.floor((end-beg)/step).astype(int)
    prefixes = [out_prefix+str(i) for i in range(numFrames+1)]
    start_files = [p+'.pdb' for p in prefixes]
    tpr_files = [p+'.tpr' for p in prefixes]
    result_prefixes = ['umbrella/'+p for p in prefixes]
    trjs = [ p+'.xtc' for p in result_prefixes ]
    xvgs = [ p+'.xvg' for p in result_prefixes ]
    os.makedirs('umbrella')
    for i in range(len(prefixes)):
        grompp(mdp_file,start_files[i],top_file,ndx_file,tpr_files[i])
        mdrun(tpr_files[i],result_prefixes[i])
        extract_position_from_traj_using_index(trjs[i],start_files[i],\
            ndx_file,ndx,xvgs[i])

def calcWork3D_Trap(x,f):
    dx=(x[1:,:]-x[:-1,:])
    f_avg = 0.5*(f[1:,:]+f[:-1,:])
    num_rows = dx.shape[0]
    norms = [np.linalg.norm(dx[i,:]) for i in range(num_rows)]
    ds = np.array(norms)  # convert to Angstroms
    dots = [np.dot(f_avg[i,:],dx[i,:]) for i in range(num_rows)]
    integrand = -np.array(dots)
    data_dict = {'s':np.cumsum(ds)*10, 'dW':integrand, 'work':np.cumsum(integrand)}
    return pd.DataFrame(data=data_dict)

def process_pmf_data(prefix,numFrames,k):
    prefixes = [prefix+str(i) for i in range(numFrames+1)]
    start_files = [p+'.pdb' for p in prefixes]
    result_prefixes = ['umbrella/'+p for p in prefixes]
    trjs = [ p+'.xtc' for p in result_prefixes ]
    xvgs = [ p+'.xvg' for p in result_prefixes ]
    pos_list = []
    shifts_list = []
    for i in range(len(prefixes)):
        data = read_xvg(xvgs[i])
        pos_list.append(xvg_ylabel_first(data))
        shifts_list.append(xvg_ylabel_shift(data))
    x = pd.concat(pos_list, axis=1).T.to_numpy()
    delta_x = pd.concat(shifts_list, axis=1).T.to_numpy()
    f = delta_x*k # sign is positive because of Newton's 3rd law and Hooke's
    return calcWork3D_Trap(x,f), x, f

def workBasedOnForceMovingAverage(xvg,velocity):
    # assumes comes from pullf.xvg
    time=xvg['data'][xvg['xaxis label']]
    force=xvg['data'][xvg['yaxis labels']]
    dt = time[1] - time[0] # ps
    N = int(0.1 / velocity / dt)
    #N=int(10 / dt)
    move_mean = pd.Series(force.to_numpy().flatten()).rolling(N).mean()
    integrand = move_mean * dt * velocity
    work = np.cumsum(integrand)
    return (move_mean, work, force, time)

def jarzynski(work,thermal):
    A = -thermal*np.log(np.mean(np.exp(-work/thermal),axis=0))
    return A

def normalAverage(work,thermal):
    W = np.mean(work,axis=0)
    return W

def averageWork(prefix,nums,velocity,temp,func=jarzynski):
    thermal = 8.314e-3*temp  # thermal energy KJ/mol
    num_runs = len(nums)
    pullf_names = [prefix+str(num)+'_pullf.xvg' for num in nums]
    xvgs = [read_xvg(n) for n in pullf_names]
    results = [workBasedOnForceMovingAverage(xvg,velocity) for xvg in xvgs]
    work = np.stack([res[1] for res in results])
    A = func(work,thermal)
    return A,results[0][3]

def plot_pull(pullf_file,pullx_file,coord_file,com_file,pull_vec,rate,k):
    f_xvg=read_xvg(pullf_file)
    x_xvg=read_xvg(pullx_file)
    c_xvg=read_xvg(coord_file)
    b_xvg=read_xvg(com_file)
    t=f_xvg['data']['Time (ps)'].to_numpy().astype(np.float)
    f=f_xvg['data'].to_numpy().astype(np.float)[:,1:]
    dx=f/k
    x=x_xvg['data'].to_numpy().astype(np.float)[:,1:]
    c=c_xvg['data'].to_numpy().astype(np.float)[:,1:]
    b=b_xvg['data'].to_numpy().astype(np.float)[:,1:]
    v=np.array(pull_vec)
    u=v/np.linalg.norm(v)
    s=t.reshape((t.shape[0],1))*0.01*u.reshape((1,u.shape[0]))+c[0,:]
    d = (c-b).dot(u)
    s2 = d+dx.T
    s3 = s2.reshape((t.shape[0],1))*u.reshape((1,u.shape[0]))+c[0,:]
    print(c[1500])
    s4 = c+dx*u.reshape((1,u.shape[0]))
    print(s[1500])
    print(s4[1500])
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    label = 'projection of difference between C18 and bottleneck COM on vector'
    ax1.plot(t,d,color='k',linestyle='--',label=label)
    label = 'gromacs produced "x" coordinate'
    ax1.scatter(t,x,color='b', label=label)
    label = 'infered position of virtual spring based on force'
    ax1.scatter(t,s2,color='r', label=label)
    ax1.set_xlabel('time(ps)')
    ax1.set_ylabel('x(nm)')
    ax1.legend(loc = 'best')
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    label = "Position of C18"
    ax2.scatter(c[:,0], c[:,1], c[:,2],color='r', label=label)
    label = "Position of bottleneck COM"
    ax2.scatter(b[:,0], b[:,1], b[:,2],color='k', label=label)
    label =  "Constant rate along normalized vector rooted at initial C18"
    ax2.scatter(s[:,0], s[:,1], s[:,2],color='b', label=label)
    label = "Implied virtual spring position based on gromacs 'x' and 'f'"
    ax2.scatter(s3[:,0], s3[:,1], s3[:,2],color='c',label=label)
    label = "Implied virtual spring position based on C18 and 'f'"
    ax2.scatter(s4[:,0], s4[:,1], s4[:,2],color='m',label=label)
    ax2.set_xlabel('X(nm)')
    ax2.set_ylabel('Y(nm)')
    ax2.set_zlabel('Z(nm)')
    ax2.legend(loc = 'best')
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(211)
    ax3.scatter(t,f,label="gromcs 'f' (KJ/mol/nm)",color="k")
    ax3.set_ylabel('Force(KJ/mol/nm)')
    ax3.legend(loc = 'best')
    ax4 = fig3.add_subplot(212)
    ax4.scatter(t,dx,label="implied stretch in spring (nm)")
    ax4.set_xlabel('time(ps)')
    ax4.set_ylabel('stretch(nm)')
    ax4.legend(loc = 'best')
    plt.show()

if __name__ == '__main__':
    prefix = sys.argv[1]
    numFrames = int(sys.argv[2])
    k = float(sys.argv[3])
    data, x, f = process_pmf_data(prefix,numFrames,k)
    print(data)
    print(x)
    print(f)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax1.scatter('s','dW', data=data)
    plt.ylabel('dW (kJ/mol)')
    ax2 = fig.add_subplot(212)
    ax2.scatter('s','work', data=data)
    plt.xlabel('s (A)')
    plt.ylabel('work (kJ/mol)')
    # Show 3D plots of figures
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111, projection='3d')
    a=x*10.0
    dx=(a[1:,:]-a[:-1,:])
    ax.scatter(a[:,0], a[:,1], a[:,2],alpha=1)
    ax.quiver(a[:-1,0], a[:-1,1], a[:-1,2],dx[:,0],dx[:,1],dx[:,2])
    b = 10.0*(f/k)
    ax.quiver(a[:,0], a[:,1], a[:,2],b[:,0], b[:,1], b[:,2],color='r')
    ax.set_xlabel('X(A)')
    ax.set_ylabel('Y(A)')
    ax.set_zlabel('Z(A)')
    plt.show()
