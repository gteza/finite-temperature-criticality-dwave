"""
Analysis functions for the finite-temperature 2D Ising quantum annealing manuscript.

This file contains helper functions used to perform the superspins embedding, the
custm spin-gauge transformations and the shimming proceddure.
"""

import numpy as np
import pandas as pd

### This function creates an embedding for all the superspin pairs for a 2D-squared or a 3D-cubic lattice on a pegasus graph.
### This embedding function can be used for the shimming.
def pegasus_2D_full_spins(n_rows=15, n_cols=15,MagneticField=0.,ChainConstant=-2.,missing_nodes=np.array([]),missing_edges=np.array([])):
    n=16
    N_tot=8*(n-1)*(3*n-1)
    N_units=3*(n-1)**2
    
    first_H=2*(n-1)
    first_bulk_H=4*(n-1)
    first_V=int(N_tot/2)+6*(n-1)
    first_bulk_V=first_V+6*(n-1)

    h_i={};J_ij={};chains_i=[]

    #internal couplers
    for I in range(n_rows):
        for J in range(n_cols):
            start_H=first_bulk_H + I + J*12*(n-1)
            start_V=first_bulk_V +J + I*12*(n-1)
            for k in range(3):
                start_inner_H=start_H + 4*k*(n-1)
                start_inner_V=start_V + 4*(2-k)*(n-1)
                delta=n-1
                for l in range(0,4):
                    source=start_inner_H + delta*l
                    target=start_inner_V + delta*l
                    if any(item in [source,target] for item in missing_nodes):
                        if source not in missing_nodes:
                            h_i.update({source: MagneticField})
                        if target not in missing_nodes:
                            h_i.update({target: MagneticField})
                        continue
                    else:
                        h_i.update({node: MagneticField for node in [source,target]})
                        J_ij.update({tuple(np.sort([source,target])): ChainConstant})

                        chains_i.append(np.sort([source,target]))
    chains_i=np.array(chains_i)

    #delete extra missing edges
    for edge in missing_edges:
        J_ij.pop((edge[0],edge[1]),None)

    return h_i,J_ij,chains_i

#SHIMMING
def adjust_fbos(response,fbo_i,alpha,sampler):
    dftemp=response.to_pandas_dataframe(sample_column=False)
    dftemp=dftemp.drop(['energy','num_occurrences'],axis=1)
    dftemp.columns=dftemp.columns.astype('int')
    avg_mags=dftemp.mean()
    
    new_fbo_i=fbo_i.copy()
    for qubit,mag in avg_mags.items():
        new_fbo_i[qubit]-=alpha*mag
    return new_fbo_i

#FUNCTIONS TO REPEAT EXPERIMENTS FOR SHIMMING
def do_experiment(h_i,J_ij,chains_i,exper_pars,fbo_i,sampler):
    exper_pars.update({'flux_biases': fbo_i})
    result=sampler.sample_ising(h_i,J_ij,**exper_pars)
    return result

#SPIN GAUGE TRANSFORMATION
def spin_reversal_manual(h_i,J_ij,G,chains_i,ChainConstant):
    list_flipped=[]
    rev_h_i=h_i.copy();rev_J_ij=J_ij.copy()
    
    for chain in chains_i:
        if (np.random.uniform()>.5):
            list_flipped.append(chain)
            for node in chain:
                rev_h_i[node]*=-1
                for neighbor in nx.all_neighbors(G,node):
                    link=tuple(np.sort([node,neighbor]))
                    if (J_ij[link]!=ChainConstant):
                        rev_J_ij[link]*=-1.
                        
    list_flipped=np.array(list_flipped)
    return rev_h_i,rev_J_ij,list_flipped
