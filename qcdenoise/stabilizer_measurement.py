import numpy as np
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
from sympy import I

def get_unique_operators(stabilizers=[]):
    """ strip leading sign +/- from stabilizer strings """
    operator_strings = [x[1:] for x in stabilizers]
    return list(set(operator_strings))

def sigma_prod(op_str):
    pauli_dict= {"I":"I","X":Pauli(1), "Y":Pauli(2),"Z":Pauli(3)}
    pauli_label = {Pauli(1):"X",Pauli(2):"Y",Pauli(3):"Z","I":"I"}
    op_list=list(op_str)
    if ('X' not in op_str) and ('Y' not in op_str) and ('Z' not in op_str):
        mat3='I'
        coef=np.complex(1,0)
    pauli_list = [pauli_dict[x] for x in op_list]
    coef_list = []
    while len(pauli_list)>1:
        mat1=pauli_list.pop()
        mat2=pauli_list.pop()
        if mat1==mat2:
            mat3='I'
            coef=np.complex(1,0)
        elif 'I' not in [mat1,mat2]:
            mat3=evaluate_pauli_product(mat2*mat1).args[-1]
            coef=evaluate_pauli_product(mat2*mat1).args[:-1]
            if coef==(I,):
                coef=np.complex(0,1)
            elif coef==(-1,I):
                coef=np.complex(0,-1)
            else:
                coef=np.complex(1,0)
        else:
            mat3=[x for x in [mat1,mat2] if x!='I'][0]
            coef=np.complex(1,0)
        coef_list.append(coef)
        pauli_list.append(mat3)
    return np.prod(np.asarray(coef_list)),[pauli_label[x] for x in pauli_list][0]

def gamma_prod(gamma_ops=[]):
    print('WIP!!')
    raise NotImplementedError

def build_stabilizer_meas(circ,stabilizer_str,drop_coef=True):
    ''' build a circuit block that implements the measurements in a stabilizer'''
    temp_circ = circ.copy()
    if drop_coef==True:
        stab_ops = list(stabilizer_str[1:])[::-1]
    else:
        stab_ops = list(stabilizer_str)[::-1]
    for idx in range(len(stab_ops)):
        op_str = stab_ops[idx]
        if op_str=='X':
            # measure X basis
            temp_circ.h(idx)
        elif op_str=='Y':
            # measure Y basis
            temp_circ.sdg(idx)
            temp_circ.h(idx)
        elif (op_str=='I') or (op_str=='Z'):
            temp_circ.i(idx)
    # test: add another layout of idenetity gates pading
    #temp_circ.barrier()
    #for jdx in range(len(stab_ops)):
    #    temp_circ.i(idx)
    return temp_circ
