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
    return calcWork3D_Trap(x,f)


if __name__ == '__main__':
    prefix = sys.argv[1]
    numFrames = int(sys.argv[2])
    k = float(sys.argv[3])
    data = process_pmf_data(prefix,numFrames,k)
    print(data)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.scatter('s','dW', data=data)
    plt.ylabel('dW (kJ/mol)')
    ax2 = fig.add_subplot(212)
    ax2.scatter('s','work', data=data)
    plt.xlabel('s (A)')
    plt.ylabel('work (kJ/mol)')
    plt.show()
