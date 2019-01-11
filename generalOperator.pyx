# distutils: language = c++


# __all__ = ['EnergyCurrent']

import cython
#from operator import itemgetter
#import functools as ft
#import collections

###For Block _eval_hamil
#import numpy as np
import tinyarray as ta
#from scipy.sparse import coo_matrix

###FOR _is_herm
from libc cimport math

import kwant
#from kwant.operators cimport _LocalOperator
#from defs cimport gint
#from kwant cimport operator
#from kwant.graph.defs import gint_dtype
from kwant.system import InfiniteSystem
###FOR _check_ham
from kwant._common import UserCodeError, get_parameters

from libc.stdlib cimport free, calloc, malloc
from cython.view cimport array as cvarray
from inspect import signature

import numpy as np


from libcpp.vector cimport vector


gint_dtype = np.int32
cdef extern from "defs.h":
    ctypedef signed int gint

# supported operations within the `_operate` method
ctypedef enum operation:
    MAT_ELS
    ACT




# --------  Definition of special operators --------

cdef class Density:
    """An operator for calculating general densities.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the operator. If a dict is given, it
        maps from site families to square matrices. If a function is given it
        must take the same arguments as the onsite Hamiltonian functions of the
        system.
    where : sequence of `int` or `~kwant.builder.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of integers. If a function is provided,
        it should take a single `int` or `~kwant.builder.Site` (if ``syst`` is
        a finalized builder) and return True or False.  If not provided, the
        operator will be calculated over all sites in the system.
    check_hermiticity: bool
        Check whether the provided ``onsite`` is Hermitian. If it is not
        Hermitian, then an error will be raised when the operator is
        evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `Operator.__call__` for details).

    Notes
    -----
    In general, if there is a certain "density" (e.g. charge or spin) that is
    represented by a square matrix :math:`M_i` associated with each site
    :math:`i` then an instance of this class represents the tensor
    :math:`Q_{iαβ}` which is equal to :math:`M_i` when α and β are orbitals on
    site :math:`i`, and zero otherwise.
    """

    cdef object dens

    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *, check_hermiticity=True, sum=False):
        # create instance of Onsite
        self.dens = Onsite(syst, onsite, where, check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return self.dens.bind(ops_tobebound='all', args=args, params=params)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        return self.dens(bra, ket, args, params=params)

    @cython.embedsignature
    def get_sitechains(self):
        return [list(path) for path in self.dens.sitechains]


cdef class Current:
    r"""An operator for calculating general currents.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    where : sequence of pairs of `int` or `~kwant.builder.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of pairs of integers. If a function is
        provided, it should take a pair of integers or a pair of
        `~kwant.builder.Site` (if ``syst`` is a finalized builder) and return
        True or False.  If not provided, the operator will be calculated over
        all hoppings in the system.
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `Operator.__call__` for details).

    Notes
    -----
    In general, if there is a certain "density" (e.g. charge or spin) that is
    represented by a square matrix :math:`M_i` associated with each site
    :math:`i` and :math:`H_{ij}` is the hopping Hamiltonian from site :math:`j`
    to site `i`, then an instance of this class represents the tensor
    :math:`J_{ijαβ}` which is equal to :math:`i\left[(H_{ij})^† M_i - M_i
    H_{ij}\right]` when α and β are orbitals on sites :math:`i` and :math:`j`
    respectively, and zero otherwise.

    The tensor :math:`J_{ijαβ}` will also be referred to as :math:`Q_{nαβ}`,
    where :math:`n` is the index of hopping :math:`(i, j)` in ``where``.
    """

    cdef object current

    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *,
                 check_hermiticity=True, sum=False):
        #create instances of Operator and Onsite and multiply them
        hamil = Operator(syst, syst.hamiltonian, where, withRevTerm=-1, const_fac=-1j, check_hermiticity=check_hermiticity, sum=sum)

        if onsite==1:
            self.current = hamil
        else:
            onsiteOp = Onsite(syst, onsite, where=None, check_hermiticity=check_hermiticity, sum=sum, willNotBeCalled=True)

            self.current = Op_Product(onsiteOp, hamil, withRevTerm=-1, const_fac=-1j, check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return self.current.bind(ops_tobebound='all', args=args, params=params)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        return self.current(bra, ket, args, params=params)

    def get_sitechains(self):
        return [list(path) for path in self.current.sitechains]



cdef class Source:
    """An operator for calculating general sources.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this source is
        defined. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    where : sequence of `int` or `~kwant.builder.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of integers. If a function is provided,
        it should take a single `int` or `~kwant.builder.Site` (if ``syst`` is
        a finalized builder) and return True or False.  If not provided, the
        operator will be calculated over all sites in the system.
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it is not
        Hermitian, then an error will be raised when the operator is
        evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `Operator.__call__` for details).

    Notes
    -----
    An example of a "source" is a spin torque. In general, if there is a
    certain "density" (e.g. charge or spin) that is represented by  a square
    matrix :math:`M_i` associated with each site :math:`i`, and :math:`H_{i}`
    is the onsite Hamiltonian on site site `i`, then an instance of this class
    represents the tensor :math:`Q_{iαβ}` which is equal to :math:`(H_{i})^†
    M_i - M_i H_{i}` when α and β are orbitals on site :math:`i`, and zero
    otherwise.
    """

    cdef object source

    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *, check_hermiticity=True, sum=False):

        # define onsite Hamiltonian
        def onsiteHamil(a, *args, **params):
            if type(a) == kwant.builder.Site:
                a = syst.id_by_site[a]
            assert(type(a) == int)
            return syst.hamiltonian(a, a, *args, params=params)

        hamil = Onsite(syst, onsiteHamil, where=None, withRevTerm=-1, const_fac=-1j, check_hermiticity=check_hermiticity, sum=sum, willNotBeCalled=True)

        if onsite == 1:
            self.source = self.hamil
        else:
            onsiteOp = Onsite(syst, onsite, where, check_hermiticity=check_hermiticity, sum=sum)

            self.source = Op_Product(onsiteOp, hamil, withRevTerm=-1, const_fac=-1j, check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return self.source.bind(ops_tobebound='all', args=args, params=params)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        return self.source(bra, ket, args, params=params)

    def get_sitechains(self):
        return [list(path) for path in self.source.sitechains]



cdef class ArbitHop:
    r"""An operator for calculating general ??? (I don't know how to call it --
    it is not a current).

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    where : sequence of pairs of `int` or `~kwant.builder.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of pairs of integers. If a function is
        provided, it should take a pair of integers or a pair of
        `~kwant.builder.Site` (if ``syst`` is a finalized builder) and return
        True or False.  If not provided, the operator will be calculated over
        all hoppings in the system.
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `~kwant.operator.Current.__call__` for details).

    Notes
    -----
    This operator is at the moment needed for calculating the explicite time-
    dependent part of the Coupling-Term I_c in the LeadHeatCurrent.
    In general, having an onsite operator represented by a square matrix
    :math:`M_i` associated with each site :math:`i` and :math:`O_{ij}` is
    a hopping Operator from site :math:`j` to site `i`, then an instance of
    this class represents the tensor
    :math:`J_{ijαβ}` which is equal to :math:`\left[(O_{ij})^† M_i + M_i
    O_{ij}\right]` when α and β are orbitals on sites :math:`i` and :math:`j`
    respectively, and zero otherwise.

    The tensor :math:`J_{ijαβ}` will also be referred to as :math:`Q_{nαβ}`,
    where :math:`n` is the index of hopping :math:`(i, j)` in ``where``.
    """

    cdef object arbitHop, onsite, hopping

    @cython.embedsignature
    def __init__(self, syst, onsite=1, arbit_hop_func=0, where=None, *,
                 check_hermiticity=True, sum=False, small_h=0.01):

        self.hopping = Operator(syst, arbit_hop_func, where, withRevTerm=+1,
                                const_fac=1, check_hermiticity=check_hermiticity,
                                sum=sum)
        if onsite == 1:
            self.arbitHop = self.hopping
        else:
            self.onsite = Onsite(syst, onsite, where=None, check_hermiticity=check_hermiticity, sum=sum, willNotBeCalled=True)

            self.arbitHop = Op_Product(self.onsite, self.hopping, withRevTerm=+1, const_fac=1, check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return self.arbitHop.bind(ops_tobebound='all', args=args, params=params)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        return self.arbitHop(bra, ket, args, params=params)

    def get_sitechains(self):
        return [list(path) for path in self.arbitcurrent.sitechains]


cdef class offEnergyCurrentLead:
    r"""An operator for calculating one part of the lead energy currents.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    where : stacked lists of hoppings [[intralead-hoppings],[lead-sys-hoppings]]
    ##COMMENT: check hermiticity not needed for hamiltonians!
    # check_hermiticity : bool
    #    Check whether the provided ``onsite`` is Hermitian. If it
    #    is not Hermitian, then an error will be raised when the
    #    operator is evaluated. (NOTE: Probably not necessary here.)
    sum : bool, default: true
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        Operator.__call__` for details).
        It should be true for the lead energy current, where we need the sum.

    Notes
    -----
    To be evaluated: :math:`\sum_{ijq} [ -2 Im\{ bra^\ast_{q} H_{qi} H_{ij} ket_{j}  \}]`,
    where :math:`q` is a site in the scattering region and :math:`i\neq j` are
    sites in the lead.
    Since :math:`H_{qi}` is only non-zero for 1st lead unit cell, it has to be
    in 1st lead unit cell and :math:`j` either in 1st or 2nd lead unit cell.
    """

    cdef object hamil1, hamil2, offEn

    @cython.embedsignature
    def __init__(self, syst, where, *, check_hermiticity=True, sum=True):

        self.hamil1 = Operator(syst, syst.hamiltonian, where[0], check_hermiticity=check_hermiticity, sum=sum)
        self.hamil2 = Operator(syst, syst.hamiltonian, where[1], check_hermiticity=check_hermiticity, sum=sum)

        self.offEn = Op_Product(self.hamil1, self.hamil2, withRevTerm=-1, const_fac=1j, check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return self.offEn.bind(ops_tobebound='all', args=args, params=params)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        return self.offEn(bra, ket, args, params=params)

    def get_sitechains(self):
        return [list(path) for path in self.offEn.sitechains]









### --- recursive functions ---
# the actual calculation of the matrix elements happens here


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sitePath_recFunc(complex[:] out_data, #complex* out_data,
                           int* data_count, int iSite,
                            int depth, int ket_start, int bra_start,
                            vector[int] &auxwhere_list,
                            vector[int] &wherepos_neigh,
                            vector[int] &auxpos_list,
                            # int* auxwhere_list,
                            # int* wherepos_neigh,
                            # int* auxpos_list,
                            int* x_norbs,
                            complex** opMat_array,
                            complex** opMat_rev_array,
                            complex** op_blocks,
                            complex** op_rev_blocks,
                            int * block_shapes,
                            int * ket_start_positions,
                            int withRevTerm, complex const_fac, int N_ops,
                            #int* unique_array,
                            vector[int] &unique_list,
                            complex[:] bra, complex[:] ket
                            # complex* bra, complex* ket
                           ):
    r"""
    Recursive functions for preparing the necessary matrices for a given set of
    sites, looping over all connected sites and initiate the sum over the orbitals
    """
    cdef int iNextSite, o1_a, nextSiteOffset
    cdef int num_neighbors, startindex_neighbors, auxpos_list_index
    cdef complex sumtmp
    cdef complex orbprod0, orbprod1

    #recursion end: get ket_start and initiate sum over all orbitals
    if depth == N_ops:
        # get index ket_start
        ket_start = ket_start_positions[iSite]
        # sum over all orbitals for the given set of sites
        sumtmp = 0
        for o1_a in range(x_norbs[0]):
            orbprod0 = bra[bra_start + o1_a].conjugate()
            if withRevTerm != 0:
                orbprod1 = ket[bra_start + o1_a]
            else:
                orbprod1 = 0j
            # calculate sum over all orbitals
            sumtmp = orbsums_recfunc(sumtmp,
                                     orbprod0, orbprod1, 0,
                                     bra_start, ket_start, o1_a,
                                     withRevTerm,
                                     x_norbs,
                                     N_ops,
                                     opMat_array,
                                     opMat_rev_array,
                                     bra, ket
                                     )
        out_data[data_count[0]] = const_fac * sumtmp
        data_count[0] = data_count[0] + 1

    #recursion step:
    #           loop over all 'x'-Sites, get the needed op-matrix and x_norbs,
    #           and call next recursion step
    else:
        assert depth < N_ops

        # find all connected sites with the help of the auxlists and loop over them
        auxpos_list_index = auxwhere_list[depth-1] + iSite
        startindex_neighbors = auxpos_list[auxpos_list_index]
        num_neighbors = auxpos_list[auxpos_list_index+1] - startindex_neighbors
        for i in range(num_neighbors):
            iNextSite = wherepos_neigh[startindex_neighbors + i]
            nextSiteOffset = auxwhere_list[depth]
            if not unique_list[depth]:
                opMat_array[depth] = op_blocks[nextSiteOffset + iNextSite]
                if withRevTerm:
                    opMat_rev_array[depth] = op_rev_blocks[nextSiteOffset + iNextSite]

            x_norbs[depth+1] = block_shapes[nextSiteOffset + iNextSite]

            # call the next step
            sitePath_recFunc(out_data,
                             data_count, iNextSite,
                             depth+1,
                             ket_start, bra_start,
                             auxwhere_list,
                             wherepos_neigh,
                             auxpos_list,
                             x_norbs,
                             opMat_array,
                             opMat_rev_array,
                             op_blocks,
                             op_rev_blocks,
                             block_shapes, ket_start_positions,
                             withRevTerm, const_fac,
                             N_ops, unique_list,
                             bra, ket
                            )



@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex orbsums_recfunc(complex sum, complex orbprod0, complex orbprod1,
                             int orbdepth, int bra_start, int ket_start,
                             int o_x, int withRevTerm, int* x_norbs,
                             int N_ops,
                             complex ** opMat_array,
                             complex ** opMat_rev_array,
                             complex[:] bra, complex[:] ket
                             # complex* bra, complex* ket
                             ):
    r"""
    calculate the sum over all orbitals for a given set of sites passed
    indirectly by `sitePath_recFunc`
    """

    cdef int norbs, norbs_next, o_y
    cdef complex orbprod_tmp0, orbprod_tmp1

    norbs = x_norbs[orbdepth]
    cdef int neworbdepth = orbdepth + 1
    cdef complex sumtmp = sum

    # recursion step: multiply O_xy*orbproduct(sofar), call next step
    if orbdepth < N_ops - 1:
        norbs_next = x_norbs[orbdepth+1]
        for o_y in range(norbs_next):
            orbprod_tmp0 = opMat_array[orbdepth][o_x*norbs+o_y]
            orbprod_tmp0 *= orbprod0
            # in case the reverted term is to be calculated
            if withRevTerm != 0:
                orbprod_tmp1 = opMat_rev_array[orbdepth][o_y*norbs_next+o_x]
                orbprod_tmp1 *= orbprod1

            # call next step/operator
            sumtmp = orbsums_recfunc(sumtmp, orbprod_tmp0, orbprod_tmp1,
                                     neworbdepth, bra_start, ket_start,
                                     o_y, withRevTerm,
                                     x_norbs, N_ops,
                                     opMat_array,
                                     opMat_rev_array,
                                     bra, ket
                                     )

    # recursion end: orbprod(sofar).lastOp.ket_z; and sum over the products
    else:
        assert orbdepth == N_ops - 1

        norbs_next = x_norbs[orbdepth+1]
        for o_y in range(norbs_next):
            orbprod_tmp0 = opMat_array[orbdepth][o_x*norbs+o_y] * ket[ket_start+o_y]
            orbprod_tmp0 *= orbprod0

            sumtmp += orbprod_tmp0

            # in case the reverted term is to be calculated
            if withRevTerm != 0:
                orbprod_tmp1 = bra[ket_start+o_y].conjugate()  \
                              * opMat_rev_array[orbdepth][o_y*norbs_next+o_x]
                orbprod_tmp1 *= orbprod1
                sumtmp += withRevTerm * orbprod_tmp1


        # # in case the reverted term is to be calculated
        # if withRevTerm != 0:
        #     orbprod_tmp1 = bra[ket_start+o_x].conjugate()
        #     orbprod_tmp1 *= orbprod1
        #     sumtmp += withRevTerm * orbprod_tmp1

    return sumtmp


#recursive function to get all combinations of sites of the sums
cdef sitechains_recfunc(object sitechain_data, object sitechain_aux, int iSite,
                        int depth,
                        int[:,:] where,
                        vector[int] &auxwhere_list,
                        vector[int] &auxpos_list,
                        vector[int] &wherepos_neigh,
                        # int[:] auxwhere_list,
                        # int[:] auxpos_list,
                        # int[:] wherepos_neigh,
                        int N_ops):
    """
    Recursive function to find all possible paths.
    This recursive function is the heart of the method
    `_create_sitechains_output(..)`.
    """
    # add the next site to the chain
    cdef int offset = auxwhere_list[depth-1]
    cdef int nextSite = where[offset + iSite][1]
    sitechain_aux[depth] = nextSite

    cdef int i, auxpos_list_index, startindex_neighbors, num_neighbors, iNextSite
    #recursion end: copy sitechain_aux to sitechain_data
    if depth == N_ops:
        sitechain_data.append(sitechain_aux[:]) #sliced for copy, avoid reference
    #recursion step: find neighbor and call the next step
    else:
        assert depth < (N_ops)
        auxpos_list_index = offset + iSite
        startindex_neighbors = auxpos_list[auxpos_list_index]
        num_neighbors = auxpos_list[auxpos_list_index+1] - startindex_neighbors
        for i in range(num_neighbors):
            iNextSite = wherepos_neigh[startindex_neighbors + i]
            sitechains_recfunc(sitechain_data, sitechain_aux, iNextSite, depth+1,
                               where,
                               auxwhere_list,
                               auxpos_list,
                               wherepos_neigh,
                               N_ops)
    # end recursive function


def _create_sitechains_output(where, auxwhere_list, auxpos_list, wherepos_neigh, N_ops):
    r"""
    Returns a list, which has the same structure as the Operator
    output_data and which can therefore be used to link the output to a path
    of sites.
    The structure of this function is very similar to sitePath_recFunc with the
    difference that here, only the site-chains are stored and that it is called
    only in the `Op_Product.__init__()` instead of `Operator._operate()`.
    """

    sitechain_data = []
    sitechain_aux = [None] * (N_ops + 1)
    # loop over all 'a'-Sites and initialize first step (for 'b'-Sites)
    for ia in range(auxwhere_list[1]):
        sitechain_aux[0] = where[ia][0]
        # call recursive function
        sitechains_recfunc(sitechain_data, sitechain_aux, ia, 1,
                           where, auxwhere_list, auxpos_list, wherepos_neigh, N_ops)

    if sitechain_data == []:
        raise UserCodeError('No path along the given where\'s of the different operators is found! Calculation is aborted!')
    return np.asarray(sitechain_data, dtype=gint_dtype)





################ Generic Utility functions

def _where_to_dict_where(syst, where):
    r"""
    Returns a dictionary that helps finding all possible site-paths which are
    needed for the product of operators.

    Parameters
    ----------
    syst : `~kwant.system.System`
    where :  list of hoppings [(a_1, b_1), (a_2, b_2), ..., (a_n, b_n)],
        with a_i, b_i, being Sites of the finalized builder (ie. of type int) or
        unfinalized builder (ie. instance of kwant.builder.Site).
    Returns
    -------
    dict_where : dict, whose keys are site-IDs and whose values are lists of indeces,
        which indicate the positions of the hoppings in `where`, whose 0-th
        hopping element is that site.
        Otherwise said: Be site1 a site_ID, then dict_where[site1] returns a
        list with all the positions of the hoppings in where, of the format
        (site_1, site_x).
        Therefore, we have:
        where[idx][0] == site1 for idx in dict_where[site1]

    """

    assert(len(where[0]) == 2)

    dict_where = {}
    for index, hop in enumerate(where):
        # if sites are kwant.builder.Site objects, convert them to Site-IDs (int)
        if isinstance(hop[0], kwant.builder.Site):
            site_int = syst.id_by_site[hop[0]]
        else:
            site_int = hop[0]
        ### creation of dict:
        # if siteID is already a key -> append position in where
        if site_int in dict_where.keys():
            dict_where[site_int].append(index)
        else:  # create new key and its value is the index of that site in where
            dict_where[site_int] = [index]

    return dict_where




def _dict_and_wherelist_to_auxlists(wherelist, dict_where):
    """
    Creates the auxiliary listes `wherepos_neigh` and `auxpos_list` from the
    dict created in `_where_to_dict_where(..)` and the where_list which
    correspond to it. With the help of theses auxiliary lists, the path finding
    becomes trivial.
    It has to be called for each product between 2 operators.

    Parameters
    ----------
    wherelist: list of hoppings of the form:
        [ [(a1,b1), ... a-b-hoppings], [(bx,c1), ... b-c-hoppings]]
    dict_list: list of dictionary whose keys are the Sites of one operator and
        its data is a list of all the neighbors (their position in the
        wherelist) of that "key"-Site for the next hopping operator

    Returns
    -------
    wherepos_neigh: values of dict_where, i.e. the position in wherelist of all
        the connected b-c-hoppings for a given a-b-hopping (b-sites are the same)
    auxpos_list: auxiliary list to help find the correct elements in wherepos_neigh.
        It yieds the starting positions in wherepos_neigh for the corresponding
        hopping in wherelist.
    """
    # assert that wherelist is a list of two where, i.e. of 2 lists of hoppings
    assert len(wherelist) == 2
    # assert that wherelist contains hoppings
    assert len(wherelist[0][0]) == 2

    auxpos_list = []
    wherepos_neigh = []
    new_pos = 0

    ab_hops = wherelist[0]
    for hop in ab_hops:
        # try to find connected hopping
        try:
            neighlist = dict_where[hop[1]]
        except KeyError:
            # Unfortunately, the following error message cannot tell which operator
            print('Site ', hop[1], 'has no matching site in next where!')
            neighlist = []
        # append list of neighbors to wherepos_neigh
        wherepos_neigh.extend(neighlist)
        num_neigh = len(neighlist)
        # append starting position in wherepos_neigh to auxpos_list
        auxpos_list.append(new_pos)
        new_pos += num_neigh
    #append len of wherepos_neigh to auxpos_list as last element
    auxpos_list.append(new_pos)

    return np.asarray(wherepos_neigh, dtype=gint_dtype), np.asarray(auxpos_list, dtype=gint_dtype)



def flatten_2d_lists_with_bookkeeping(twodim_list):
    # assert()
    value_list = []
    auxlist = [0] * (len(twodim_list)+1)

    count = 0
    for i, lst in enumerate(twodim_list):
        for value in lst:
            value_list.append(value)
            count += 1
        auxlist[i+1] = count

    return value_list, auxlist



cdef class Onsite(Operator):
    r"""
    An operator for calculating the matrix elements of an onsite operator (some
    kind of density) and/or for using it to multiply with other operators.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    opfunc : scalar, square matrix, dict or callable
        If a dict is given, it maps from site families to square matrices
        (scalars are allowed if the site family has 1 orbital per site).
        If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    where : sequence of `int` or `~kwant.builder.Site`, or callable.
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of integers. If a function is provided,
        it should take a single `int` or `~kwant.builder.Site` (if ``syst`` is
        a finalized builder) and return True or False.  If not provided, the
        operator will be calculated over all sites in the system.
    Optional
    withRevTerm : denotes if the complex conjugate is to be calculated (~reverting
        the order of the operators). It should be either +1, 0 or -1, where the
        sign denotes if the complex conjugate is added or substracted.
        ## COMMENT: Might not make sense for an onsite operator ##
    const_fac : a constant factor which is multiplied to the result of the operator
    check_hermiticity : bool, default False
        Check whether the provided ``opfunc`` is Hermitian. If it is not
        Hermitian, then an error will be raised when the operator is
        evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned for each site.
    willNotBeCalled : bool, default: False
        In case the operator is only used to be part of a product, but will not
        be called individually, this option can be used to facilitate the __init__
        of the operator product ``Op_Product`` (e.g. no path finding might be
        required). It can only increase the efficiency in the case `where==None`
        (all Sites of the system are used).


    Notes
    -----
    In general, if there is a certain "density" (e.g. charge or spin) that is
    represented by a square matrix :math:`M_i` associated with each site
    :math:`i` then an instance of this class represents the tensor
    :math:`Q_{iαβ}` which is equal to :math:`M_i` when α and β are orbitals on
    site :math:`i`, and zero otherwise.
    """

    @cython.embedsignature
    def __init__(self, syst, opfunc, where, withRevTerm=0, const_fac=1, *, check_hermiticity=False, sum=False, willNotBeCalled=False):
        # store onsite-related class variales
        self._isonsite = np.asarray([1],dtype=gint_dtype)
        self._onsite_params_info = [None]
        onsite, self._onsite_params_info[0] = kwant.operator._normalize_onsite(syst, opfunc, check_hermiticity)

        # willNotBeCalled to improve efficiency in case of an operator product
        # it is only important if where == None
        self.willNotBeCalled = willNotBeCalled and (not where)
        if self.willNotBeCalled:
            where_fakehops = None
        else:
            _where = kwant.operator._normalize_site_where(syst, where)
            _where = _where.reshape(-1)
            where_fakehops = np.asarray([(site,site) for site in _where],dtype=gint_dtype)

        # The remaining class variables are stored by Operator.__init__
        super().__init__(syst, onsite, where_fakehops, withRevTerm, const_fac,
                         check_hermiticity=check_hermiticity, sum=sum)


cdef class Op_Product(Operator):
    r"""
    An operator for calculating the product of an arbitrary amount of operators,
    which are either onsite operators, :math:`M_i`, or hopping operators,
    :math:`O_{ij}`, where :math:`i` and :math:`j` are sites (possibly :math:`i=j`).

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See `Operator.__call__` for details.

    Here, 4 different cases are considered:
        - Operators were not callable and will *not* be callable after product
        - Operators were not callable and will be callable after product
        - Some Operators were not callable other were callable. Product callable
        - None of the operators was not callable. Product callable

    Parameters
    ----------
    ops : a list of of operators to be multiplied. In most cases, these Operators
          are instances of either `Operator` or `Onsite`.
    Optional
    withRevTerm: denotes if the term with reversed order of the operators is to
        be calculated (~complex conjugate). It should be either +1, 0 or -1,
        where the sign denotes if the reverted term is added or substracted.
    const_fac: a constant factor which is multiplied to the result of the operator
    check_hermiticity : bool, default: False
        Check whether each single operator is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned for each site combination.
    willNotBeCalled : bool, default: False
        In case the operator is only used to be part of a product, but will not
        be called individually, this option can be used to facilitate the __init__
        of the operator product ``Op_Product`` (e.g. no path finding might be
        required). However, this option can only increase the efficiency if
        every single operator in `ops` has `willNotBeCalled==True`.

    Notes
    -----
    Examples, where products of operators are needed are general currents,
    :math:`i\left[(H_{ij})^† M_i - M_iH_{ij}\right]`, where :math:`M_i` is
    a square matrix that represents a certain "density" (e.g. spin)
    and :math:`H_{ij}` is the hopping Hamiltonian from site :math:`j` to
    site `i`, or the energy current into(??) a lead given by
    :math:`\sum_{ijq} [ -2 Im\{ bra^\ast_{q} H_{qi} H_{ij} ket_{j}  \}]`,
    where :math:`q` is a site in the scattering region and :math:`i`, :math:`j`
    are sites in the lead.
    The calculation for the matrix elements of the product of operators is
    similar to the kwant.operators, with the difference that now
    lists of some objects are needed. The arbitrary amount of operators is handled
    by recursive functions, e.g. to calculate the sum over the orbitals.

    Implementation Notes
    --------------------
    For efficiency reasons, the recursive functions to calculate the product
    over an arbitrarz amount of operators are cdef-functions.
    For some reasons, the passed lists are most efficient if they are c-arrays
    (MemViews slow down the calculation by a factor of ~3 compared to c-arrays).

    The structure of the code is atm "grown" historically, i.e. from python objects
    to MemViews, to c-arrays, which is why there are still some redundant steps,
    e.g. creating first MemViews which are later copied to c-arrays, or the
    creation of the dict to find connected sites.

    Unfortunately, variable names do not follow a strict convention. Some of the
    names are not precise or even misleading. (To be fixed.)
    """

    def __init__(self, *ops, withRevTerm=0, const_fac=1, check_hermiticity=False,
                 sum=False, willNotBeCalled=False, in_wherepos_neigh_list=-1):
        assert len(ops) > 1
        cdef int i,j,k
        cdef int veclen
        # assert that system is the same for all operators
        for i in range(1, len(ops)):
            assert ops[0].syst == ops[i].syst
        # assert that in_wherepos_neigh has the right structure, if it is not
        # the default value
        if in_wherepos_neigh_list != -1:
            # for each product, we need a wherepos_neigh
            assert len(in_wherepos_neigh_list) == len(ops) - 1
            # data have to be site-IDs, ie integers
            assert type(in_wherepos_neigh_list[0][0][0]) == int
            for i in range(len(ops)-1):
                # each hopping needs to know its connected hoppings in the
                # next operator where
                veclen = len(ops[i].vecauxwhere_list)
                assert len(in_wherepos_neigh_list[i]) == ops[i].vecauxwhere_list[veclen-1] -  ops[i].vecauxwhere_list[veclen-2]
        # store class variables
        self.syst = ops[0].syst
        self._site_ranges = ops[0]._site_ranges
        self.check_hermiticity = check_hermiticity
        self.sum = sum
        self.withRevTerm = withRevTerm
        self.const_fac = const_fac

        # total amount of operators (given ops could be already products of operators)
        self.N_ops = 0
        for op in ops:
            self.N_ops += op.N_ops

        # copy data from ops to product operator
        self.oplist = [None] * self.N_ops
        self._onsite_params_info = [None] * self.N_ops
        self._isonsite = np.zeros(self.N_ops, dtype=gint_dtype)
        # self.unique_list = np.zeros(self.N_ops, dtype=gint_dtype)
        self.unique_vector.resize(self.N_ops, 0)
        self._bound_operator_list = [None] * self.N_ops
        self._bound_operator_rev_list = [None] * self.N_ops
        cdef int op_idx = 0
        for i, op in enumerate(ops):
            # any operator in ops can already be a product of many operators
            for j in range(len(op.oplist)):
                assert len(op.oplist) == len(op._onsite_params_info)
                assert len(op.oplist) == len(op.unique_vector)
                assert len(op.oplist) == len(op._isonsite)
                assert len(op.oplist) == len(op._bound_operator_list)
                assert len(op.oplist) == len(op._bound_operator_rev_list)
                self.oplist[op_idx] = op.oplist[j]
                self._onsite_params_info[op_idx] = op._onsite_params_info[j]
                self._isonsite[op_idx] = op._isonsite[j]
                # self.unique_list[op_idx] = op.unique_list[j]
                self.unique_vector[op_idx] = op.unique_vector[j]
                self._bound_operator_list[op_idx] = op._bound_operator_list[j]
                self._bound_operator_rev_list[op_idx] = op._bound_operator_rev_list[j]
                op_idx += 1

        # willNotBeCalled only relevant if all ops willNotBeCalled
        self.willNotBeCalled = willNotBeCalled
        num_UnCallableOps  = 0
        for op in ops:
            self.willNotBeCalled = self.willNotBeCalled and op.willNotBeCalled
            if op.willNotBeCalled:
                num_UnCallableOps += 1

        # at the moment, we can have arbitrary numbers of Uncallable operators
        # multiplied with each other and an arbitrary amount of Callable operators
        # multiplied with each other, but if there is a mixture between callable a
        # and uncallable operators, there can only be 1 of each
        if num_UnCallableOps != 0 and num_UnCallableOps != len(ops):
            assert len(ops) == 2

        if self.willNotBeCalled:
            print("Initialize product of uncallable operators. The product is also not callable.")
            # we already have all objects that we need if this operator is
            # not to be called
            return


        ### product-operator can be called -> we need:
        ### where_flat, auxlists, sitechains, output_length and dict_lists

        cdef int lstart, lend, rstart, rend
        cdef int dict_idx = 0

        cdef int start = 0
        cdef int end = 0
        cdef int aux_idx = 1
        cdef int offset = 0
        cdef int num_added_fakehops
        cdef int auxpos_list_len, auxpos_value_offset, auxpos_idx
        cdef int scndtolast
        # self.vecauxwhere_list = np.zeros(self.N_ops+1, dtype=gint_dtype)
        self.vecauxwhere_list.resize(self.N_ops+1, 0)

        # initial operators not to be called but product is to be called
        if ops[0].willNotBeCalled and ops[1].willNotBeCalled:
            print("Initialize product of uncallable operators. The product is callable.")
            _allSites = kwant.operator._normalize_site_where(self.syst, None)
            # atm, even for onsite operators, the elements of `where` are
            #(fake)hoppings (from one site to the same site)
            _fakehops = [(site[0],site[0]) for site in _allSites]
            self.where_flat = np.asarray(_fakehops * self.N_ops, dtype=gint_dtype)
            self.Nhops_tot = self.N_ops*len(_fakehops)


            ### The following lists are auxiliary lists to find the correct paths.
            ### In this case, they are trivial and no path finding algorithm has
            ### to be used.

            # self.vecauxwhere_list
            # self.vecauxwhere_list = np.asarray(
            #             range(0,self.Nhops_tot+1,len(_fakehops)),
            #             dtype=gint_dtype)
            self.vecauxwhere_list = range(0,self.Nhops_tot+1,len(_fakehops))

            # self.wherepos_neigh
            # self.wherepos_neigh = np.asarray(list(range(len(_fakehops)))*(self.N_ops-1), dtype=gint_dtype)
            self.vecwherepos_neigh = list(range(len(_fakehops)))*(self.N_ops-1)
            # self.auxpos_list
            # self.auxpos_list = np.asarray(list(range(0,len(_fakehops)*(self.N_ops-1)+1)), dtype=gint_dtype)
            self.vecauxpos_list = range(0,len(_fakehops)*(self.N_ops-1)+1)
            # self.sitechains
            self.sitechains = np.asarray([range(0,len(_fakehops))], dtype=gint_dtype)

        # one initial operator not to be called, the other one is callable
        elif ops[0].willNotBeCalled != ops[1].willNotBeCalled:
            #TODO: have less in distinguishing if-else statements
            print("Initialize product of one callable operator and one uncallable operator.")
            if ops[0].willNotBeCalled:
                opCallable = ops[1]
                opNotCallable = ops[0]
                # information for where_flat, which is generated after this `if`
                hop_el = 0
                start = 0
                # end = opCallable.vecauxwhere_list[1]
                end = opCallable.vecauxwhere_list[1]

                offset_uncallable = 0
                offset_callable = opNotCallable.N_ops * (end-start)
                num_added_fakehops = (end-start) * opNotCallable.N_ops


                ### no path finding algorithm has to be applied in this case
                ### since one of the operators is `willNotBeCalled`
                # self.whereaux_list
                for i in range(opNotCallable.N_ops):
                    self.vecauxwhere_list[i] = (end-start) * i
                for i in range(opCallable.N_ops+1):
                    j = i+opNotCallable.N_ops
                    self.vecauxwhere_list[j] = \
                     opCallable.vecauxwhere_list[i] + num_added_fakehops

                # self.wherepos_neigh
                # self.wherepos_neigh = np.asarray(
                         # list(range(end-start))*opNotCallable.N_ops+list(opCallable.wherepos_neigh),
                         # dtype=gint_dtype)

                self.vecwherepos_neigh = list(range(end-start)) * opNotCallable.N_ops  + list(opCallable.vecwherepos_neigh)

                # self.auxpos_list
                if len(opCallable.vecauxpos_list) == 0:
                    auxpos_list_len = num_added_fakehops + 1  #last element is length of wherepos_neigh
                else:
                    auxpos_list_len = num_added_fakehops + len(opCallable.vecauxpos_list)
                # self.vecauxpos_list = np.zeros(auxpos_list_len, dtype=gint_dtype)
                self.vecauxpos_list.resize(auxpos_list_len, 0)

                #add additional absolute 'neigh'-positions in wherepos_neigh
                for i in range(num_added_fakehops):
                    self.vecauxpos_list[i] = i
                # copy old data
                for i in range(len(opCallable.vecauxpos_list)):
                    j = i+num_added_fakehops
                    self.vecauxpos_list[j] = \
                                 opCallable.vecauxpos_list[i] + num_added_fakehops
                if len(opCallable.vecauxpos_list) == 0:
                    self.vecauxpos_list[num_added_fakehops] = num_added_fakehops

            elif ops[1].willNotBeCalled:
                opCallable = ops[0]
                opNotCallable = ops[1]

                # information for where_flat, which is generated after this `if`
                hop_el = 1
                veclen = len(opCallable.vecauxwhere_list)
                start = opCallable.vecauxwhere_list[veclen-2]
                end = opCallable.vecauxwhere_list[veclen-1]

                offset_uncallable = opCallable.Nhops_tot
                offset_callable = 0
                num_added_fakehops = (end-start) * opNotCallable.N_ops

                ### no path finding algorithm has to be applied in this case
                ### since one of the operators is `willNotBeCalled`
                # self.whereaux_list
                for i in range(opCallable.N_ops):
                    self.vecauxwhere_list[i] = opCallable.vecauxwhere_list[i]
                for i in range(opNotCallable.N_ops+1):
                    j = i+opCallable.N_ops
                    self.vecauxwhere_list[j] = \
                      (end-start) * i + opCallable.vecauxwhere_list[veclen-1]

                # self.wherepos_neigh
                # self.wherepos_neigh = np.asarray(
                #          list(opCallable.wherepos_neigh)+
                #          list(range(end-start))*(opNotCallable.N_ops),
                #          dtype=gint_dtype)

                self.vecwherepos_neigh = list(opCallable.vecwherepos_neigh) + \
                                list(range(end-start)) * (opNotCallable.N_ops)

                # self.auxpos_list
                if len(opCallable.vecauxpos_list) == 0:
                    auxpos_list_len = num_added_fakehops + 1  #last element is length of wherepos_neigh
                else:
                    auxpos_list_len = num_added_fakehops + len(opCallable.vecauxpos_list)
                # self.auxpos_list = np.zeros(auxpos_list_len, dtype=gint_dtype)
                self.vecauxpos_list.resize(auxpos_list_len, 0)
                # copy old data
                for i in range(len(opCallable.vecauxpos_list)):
                    self.vecauxpos_list[i] = opCallable.vecauxpos_list[i]
                # check for cases in which 'opCallable.vecauxpos_list' does not exist
                if len(opCallable.vecauxpos_list) == 0:
                    offset_idx = 0
                    offset_value = 0
                else:
                    offset_idx = len(opCallable.vecauxpos_list)
                    offset_value = len(opCallable.vecwherepos_neigh) + 1
                #add additional absolute 'neigh'-positions in wherepos_neigh
                for i in range(num_added_fakehops):
                    j = i+offset_idx
                    self.vecauxpos_list[j] = i + offset_value
                if len(opCallable.vecauxpos_list) == 0:
                    self.vecauxpos_list[num_added_fakehops] = num_added_fakehops

            else:
                raise ValueError("The wrong operator multiplication option was chosen by the code.")

            N_onsites = opNotCallable.N_ops
            self.Nhops_tot = opCallable.Nhops_tot + num_added_fakehops

            # self.where_flat
            self.where_flat = np.zeros((self.Nhops_tot,2), dtype=gint_dtype)
            _onsites_where = [(hop[hop_el],hop[hop_el])
                              for hop in opCallable.where_flat[start:end]] * N_onsites
                # add new fakehoppings
            for i in range( num_added_fakehops ):
                for j in range(2):
                    k = offset_uncallable+i
                    self.where_flat[k][j] = _onsites_where[i][j]
                # copy old hoppings
            for i in range(opCallable.Nhops_tot):
                for j in range(2):
                    k = i+offset_callable
                    self.where_flat[k,j] = opCallable.where_flat[i,j]

            # self.sitechains
            self.sitechains = opCallable.sitechains



        elif not ops[0].willNotBeCalled and not ops[1].willNotBeCalled:
            print("Initialize product of callable operators.")
            self.Nhops_tot = 0
            for op in ops:
                self.Nhops_tot += op.Nhops_tot
            # self.where_flat and self.vecauxwhere_list
            self.where_flat = np.zeros((self.Nhops_tot,2), dtype=gint_dtype)
            for op in ops:
                end += op.Nhops_tot
                for i in range(end-start):
                    for j in range(2):
                        self.where_flat[start+i][j] = op.where_flat[i][j]
                for where_offset in op.vecauxwhere_list[1:]:
                    self.vecauxwhere_list[aux_idx] = where_offset + start
                    aux_idx += 1
                start = end

            # self.wherepos_neigh and self.auxpos_list
            _list_wherepos_neigh = []
            # self.auxpos_list = np.zeros(self.vecauxwhere_list[-2]+1, dtype = gint_dtype)
            scndtolast = len(self.vecauxwhere_list)-2
            self.vecauxpos_list.resize(self.vecauxwhere_list[scndtolast]+1)
            auxpos_value_offset = 0
            auxpos_idx = 0
            for i in range(len(ops)-1):
                #copy old auxlists of i-th operator
                _list_wherepos_neigh = _list_wherepos_neigh + list(ops[i].vecwherepos_neigh)
                for j, value in enumerate(ops[i].vecauxpos_list[1:]):
                    auxpos_idx += 1
                    self.vecauxpos_list[auxpos_idx] = value + auxpos_value_offset

                auxpos_value_offset += len(ops[i].vecwherepos_neigh)

                # check if wherepos_neigh is given, if not find all connections
                if in_wherepos_neigh_list != -1:
                    _new_wherepos_neigh, _new_auxpos_list = flatten_2d_lists_with_bookkeeping(in_wherepos_neigh_list[i])
                else:
                    # create new dict as tool for creating the new auxlists,
                    # wherepos_neigh and auxpos_list, at the product interface
                    veclen = len(ops[i].vecauxwhere_list)
                    lstart = ops[i].vecauxwhere_list[veclen-2]
                    lend = ops[i].vecauxwhere_list[veclen-1]
                    rstart = ops[i+1].vecauxwhere_list[0]  # == 0
                    rend = ops[i+1].vecauxwhere_list[1]
                    # which hoppings are needed
                    _where_siteoverlap = [ops[i].where_flat[lstart:lend], ops[i+1].where_flat[rstart:rend]]
                    ### Major part of path finding:
                    # create dict for path finding
                    _newdict = _where_to_dict_where(self.syst, ops[i+1].where_flat[rstart:rend])
                    # use dict to create new partial auxlists
                    _new_wherepos_neigh, _new_auxpos_list = _dict_and_wherelist_to_auxlists(_where_siteoverlap, _newdict)

                # copy partial auxlists to auxlists
                _list_wherepos_neigh = _list_wherepos_neigh + list(_new_wherepos_neigh)
                for j, value in enumerate(_new_auxpos_list[1:]):
                    auxpos_idx += 1
                    self.vecauxpos_list[auxpos_idx] = value + auxpos_value_offset

                auxpos_value_offset += len(_new_wherepos_neigh)

            # copy of old auxlists of last factor
            _list_wherepos_neigh = _list_wherepos_neigh + list(ops[-1].vecwherepos_neigh)
            for j, value in enumerate(ops[-1].vecauxpos_list[1:]):
                auxpos_idx += 1
                self.vecauxpos_list[auxpos_idx] = value + auxpos_value_offset
            self.vecwherepos_neigh.resize(len(_list_wherepos_neigh))
            # self.wherepos_neigh = np.asarray(_list_wherepos_neigh, dtype=gint_dtype)
            self.vecwherepos_neigh = np.asarray(_list_wherepos_neigh, dtype=gint_dtype)

            # self.sitechains
            self.sitechains = _create_sitechains_output(self.where_flat,
                                                        self.vecauxwhere_list,
                                                        self.vecauxpos_list,
                                                        self.vecwherepos_neigh,
                                                        self.N_ops)
        else:
            raise ValueError("Cannot find appropriate oprtion for the operator multiplication.")

        # self.out_data_length is in all cases the same if product is callable
        self.out_data_length = len(self.sitechains)




cdef class Operator:
    r"""
    An operator for calculating the matrix elements of a hopping operator and/or
    for using it to multiply with other operators.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See `Operator.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    opfunc : callable
        The function must take the same arguments as the Hamiltonian functions
        of the system. ##Not sure if this statement is correct.##
        ## COMMENT: compared to Onsite, only callable is allowed here. But dicts,
        etc. might be ok, too? ##
    in_where : sequence of pairs of `int` or `~kwant.builder.Site`, or callable
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of pairs of integers. If a function is
        provided, it should take a pair of integers or a pair of
        `~kwant.builder.Site` (if ``syst`` is a finalized builder) and return
        True or False.  If not provided, the operator will be calculated over
        all hoppings in the system.
    Optional
    withRevTerm : denotes if the complex conjugate is to be calculated (~reverting
        the order of the operators). It should be either +1, 0 or -1, where the
        sign denotes if the complex conjugate is added or substracted.
    const_fac : a constant factor which is multiplied to the result of the operator
    check_hermiticity : bool, default False
        Check whether the provided ``opfunc`` is Hermitian. If it is not
        Hermitian, then an error will be raised when the operator is
        evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned for each site.

    Notes
    -----
    This class is mostly needed for the particle currents, or to multiply it
    with other operators, e.g. to get the spin current.
    """

    # mostly input parametes
    cdef public object syst, oplist
    cdef public int check_hermiticity, sum, withRevTerm
    cdef public complex const_fac
    # mostly needed in case of products of operators
    cdef public int N_ops, Nhops_tot, out_data_length
    cdef public int[:, :] sitechains
    # flattened where list of hoppings
    cdef public int[:, :]  where_flat
    # 1d-auxlists, mostly for products of operators by helping in path finding
    # cdef public int[:] auxwhere_list, wherepos_neigh, auxpos_list
    cdef public vector[int] vecauxwhere_list, vecwherepos_neigh, vecauxpos_list
    # in case of bound operators
    cdef public object _bound_operator_list, _bound_operator_rev_list, _onsite_params_info
    # in case of unique operators
    cdef public int[:] _isonsite #, unique_list
    cdef public vector[int] unique_vector
    # site information for _get_orbs
    cdef public int[:, :] _site_ranges
    # for onsite operators to avoid unnecessary path-finding
    cdef public int willNotBeCalled



    # cdef int * pointerauxwhere_list
    # cdef int * pointerwherepos_neigh
    # cdef int * pointerauxpos_list
    # cdef complex * pointerout_data

    # cdef complex * pointerbra
    # cdef complex * pointerket
    # cdef int * x_norbs
    # cdef int * unique_onsite
    # cdef complex ** M_onsite
    # cdef complex ** hopfunc
    # cdef public int numbercalls

    @cython.embedsignature
    def __init__(self, syst, opfunc, in_where=None, withRevTerm=0, const_fac=1, *, check_hermiticity=False, sum=False):
        # store passed class variables
        self.syst = syst
        self.oplist = [opfunc]
        self.withRevTerm = withRevTerm
        self.const_fac = const_fac
        self.check_hermiticity = check_hermiticity
        self.sum = sum

        # check if __init__ is called by onsite.__init__
        try:
            if self._isonsite[0]:
                pass
        except AttributeError:
            # else: init onsite variables to Default
            self._isonsite = np.zeros(1,dtype=gint_dtype)
            self._onsite_params_info = [None]
            self.willNotBeCalled = False

        # Store additional information in class variables
        self.N_ops = 1
        self.unique_vector.push_back(not callable(opfunc))
        # self.unique_list = np.asarray([not callable(opfunc)],dtype=gint_dtype)
        self._site_ranges = np.asarray(syst.site_ranges, dtype=gint_dtype)
        self._bound_operator_list = [None]
        self._bound_operator_rev_list = [None]


        cdef int i
        # where and its auxlists are not needed if this operator is not being called
        # (can only be the case for onsite operators)
        if self.willNotBeCalled:
            self.where_flat = None
            self.Nhops_tot = 0
        else:
            # check if where site are already integers instead of kwant.Sites, because
            # kwant.operator._normalize_hopping_where throws an error in that case
            # if not callable(in_where) and not in_where == None and
            try:
                if type(in_where[0][0]) == gint_dtype:
                    print('transforming SiteIDs to kwant.Sites')
                    assert len(in_where[0]) == 2
                    # if type(in_where[0][0]) == int:
                    _dummy_where = [None] * len(in_where)
                    _dummy_where = [(syst.sites[hop[0]], syst.sites[hop[1]]) for hop in in_where]
                else:
                    _dummy_where = in_where
            except TypeError:
                _dummy_where = in_where
            # normalize hoppings, get total number of hoppings
            self.where_flat = kwant.operator._normalize_hopping_where(syst, _dummy_where)
            self.Nhops_tot = len(self.where_flat)

            # The following are interesting objects only in case of products of ops.
            # They are trivial in this __init__ method but still needed for the
            # calculation.
            self.out_data_length = self.Nhops_tot
            self.sitechains = self.where_flat
            # self.sitechains = np.asarray([(hop[0], hop[1]) for hop in self.where_flat], dtype=gint_dtype)
            # self.vecauxwhere_list =  np.asarray((0, self.Nhops_tot), dtype=gint_dtype)
            self.vecauxwhere_list = {0, self.Nhops_tot}

            # self.wherepos_neigh = np.asarray([], dtype=gint_dtype)
            # self.auxpos_list = np.asarray([0], dtype=gint_dtype)

            self.vecauxpos_list.push_back(0)
            # no need for self.vecwherepos_neigh



    @cython.embedsignature
    def get_sitechains(self):
        r"""
        Returns list of all possible paths which are in the same order as
        the output_data-list.
        """
        return [list(path) for path in self.sitechains]

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        r"""
        Return the matrix elements of the operator.

        An operator ``A`` can be called like

            >>> A(psi)

        to compute the expectation value :math:`\bra{ψ} A \ket{ψ}`,
        or like

            >>> A(phi, psi)

        to compute the matrix element :math:`\bra{φ} A \ket{ψ}`.

        If ``sum=True`` was provided when constructing the operator, then
        a scalar is returned. If ``sum=False``, then a vector is returned.
        The vector is defined over the paths defined for this operator (e.g.
        sites of the system if the operator is a Density, or over the hoppings
        if it is a Current of Source, for instance).
        The returned vector is in general ordered in the same way as the
        vector returned by Operator.get_sitechains().
        For standard operators, the returned vector is defined only over
        the sites or hoppings specified, and is ordered in the same way
        as ``where``. If the keyword parameter ``where`` was not provided
        then the returned vector is ordered in the same way as the sites
        (for the `Density`) or hoppings in the graph of the system (for the
        `Current`, the `Source` and the 'CurrentWithArbitHop').

        ## COMMENT: the following is not correct at the moment for arbit ops! ##
        Alternatively stated, for an operator :math:`Q_{iαβ...ω}`, ``bra``
        :math:`φ_α` and ``ket`` :math:`ψ_ω` this computes
        :math:`q_i = ∑_{αβ..ω} φ^*_α Q_{iαβ..ω} ψ_ω` if ``self.sum`` is False,
        otherwise computes :math:`q = ∑_{iαβ} φ^*_α Q_{iαβ..ω} ψ_ω`, where
        :math:`i` runs over all sites or hoppings, and
        :math:`α`, :math:`β`, ..., :math:`ω` run over all the degrees of freedom.

        Parameters
        ----------
        bra, ket : sequence of complex
            Must have the same length as the number of orbitals
            in the system. If only one is provided, both ``bra``
            and ``ket`` are taken as equal.
        args : tuple, optional
            The arguments to pass to the system. Used to evaluate
            the ``onsite`` elements and, possibly, the system Hamiltonian.
            Mutually exclusive with 'params'.
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        `float` if ``check_hermiticity`` is True, and ``ket`` is ``None``,
        otherwise `complex`. If this operator was created with ``sum=True``,
        then a single value is returned, otherwise an array is returned.
        """

        ### Everything taken from `~kwant.opertor.__call__` if not otherwise stated

        # Sanity check to be thought of for binding only certain Operators
        # removed compared to _LocalOperator.__call__
        # if (self._bound_onsite or self._bound_hamiltonian) and (args or params):
            # raise ValueError("Extra arguments are already bound to this "
            #                  "operator. You should call this operator "
            #                  "providing neither 'args' nor 'params'.")
        if self.willNotBeCalled:
            raise ValueError("""This operator was initialized only for products
                              but not for being called directly""")
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")
        if bra is None:
            raise TypeError('bra must be an array')
        bra = np.asarray(bra, dtype=complex)
        ket = bra if ket is None else np.asarray(ket, dtype=complex)
        tot_norbs = kwant.operator._get_tot_norbs(self.syst)
        if bra.shape != (tot_norbs,):
            msg = 'vector is incorrect shape'
            msg = 'bra ' + msg if ket is None else msg
            raise ValueError(msg)
        elif ket.shape != (tot_norbs,):
            raise ValueError('ket vector is incorrect shape')

        # where changes (why?) removed compared to _LocalOperator.__call__
        # where = np.zeros((len(gen)))
        # for i in range(len(self.wherelist)):
        #     self.wherelist[i] = np.asarray(self.wherelist[i])
        #      self.wherelist[i].setflags(write=False)
        #     if self.wherelist[i].shape[1] == 1:
        #         # if `where` just contains sites, then we want a strictly 1D array
        #          self.wherelist[i] =  self.wherelist[i].reshape(-1)
        # where = np.asarray(self.where)
        # where.setflags(write=False)
        # if self.where.shape[1] == 1:
        #     # if `where` just contains sites, then we want a strictly 1D array
        #     where = where.reshape(-1)

        # Result-length is changed compared to _LocalOperator.__call__
        result = np.zeros((self.out_data_length,), dtype=complex)
        self._operate(out_data=result, bra=bra, ket=ket, args=args,
                      params=params, op=MAT_ELS)
        # if everything is Hermitian then result is real if bra == ket
        if self.check_hermiticity and bra is ket:
            result = result.real

        return np.sum(result) if self.sum else result


    @cython.embedsignature
    def bind(self, ops_tobebound=[], *, args=(), params=None):
        """Bind the given arguments to this operators in ops_tobebound.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called.
        """
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")
        # generic creation of new instance
        cls = self.__class__
        q = cls.__new__(cls)

        q.syst = self.syst
        q.oplist = self.oplist
        q.check_hermiticity = self.check_hermiticity
        q.sum = self.sum
        q.withRevTerm = self.withRevTerm
        q.const_fac = self.const_fac
        q.N_ops = self.N_ops
        q.Nhops_tot = self.Nhops_tot
        q.out_data_length = self.out_data_length
        q.sitechains = self.sitechains
        q.where_flat = self.where_flat
        # q.vecauxwhere_list = self.vecauxwhere_list
        q.vecauxwhere_list = self.vecauxwhere_list
        # q.wherepos_neigh = self.wherepos_neigh
        # q.auxpos_list = self.auxpos_list
        q.vecwherepos_neigh = self.vecwherepos_neigh
        q.vecauxpos_list = self.vecauxpos_list
        # q.unique_list = self.unique_list
        q.unique_vector = self.unique_vector
        q._site_ranges = self._site_ranges
        q._onsite_params_info = self._onsite_params_info
        q._isonsite = self._isonsite
        q._bound_operator_list = self._bound_operator_list
        q._bound_operator_rev_list = self._bound_operator_rev_list
        q.willNotBeCalled = self.willNotBeCalled

        if ops_tobebound == 'all':
            ops_tobebound = list(range(self.N_ops))
        for index in ops_tobebound:
            if callable(self.oplist[index]):
                q._bound_operator_list[index] = self._eval_operator(index, args, params)
                q._bound_operator_rev_list[index] = self._eval_operator(index,
                                                         args, params, rev=True)

        return q


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        """Do an operation with the operator.

        Parameters
        ----------
        out_data : ndarray
         Output array, zero on entry. On exit should contain the required
         data.  What this means depends on the value of `op`, as does the
         length of the array.
        bra, ket : ndarray
         Wavefunctions defined over all the orbitals of the system.
         ## COMMENT: No act implemented so far.##
         If `op` is `ACT` then `bra` is None.
        args : tuple
         The extra arguments to the Hamiltonian value functions and
         the operator ``onsite`` function. Mutually exclusive with 'params'.
        op : operation
         The operation to perform.
         `MAT_ELS`: calculate matrix elements between `bra` and `ket`
         ## COMMENT: No act implemented so far.##
         `ACT`: act on `ket` with the operator.
        params : dict, optional
         Dictionary of parameter names and their values. Mutually exclusive
         with 'args'.
        """
        cdef int i, iSite
        ### prepare matrices for operators at needed hoppings/sites
        cdef complex[:, :] _tmp_mat
        #could be done in __init__, but problems with creating global pointer!?
        # cdef int * unique_array = <int*>calloc(self.N_ops, sizeof(int))
        # for i in range(self.N_ops):
        #     unique_array[i] = self.unique_list[i]
        cdef complex **opMat_array = \
                          <complex **>calloc(self.N_ops, sizeof(complex*))
        # for reversed order of operators
        cdef complex **opMat_rev_array = \
                        <complex **>calloc(self.N_ops, sizeof(complex*))

        # cdef BlockSparseMatrix2[:] op_mat_list
        # cdef BlockSparseMatrix2[:] op_mat_rev_list
        # initializing MemView with a dummy BlockSparseMatrix
        # cdef BlockSparseMatrix2 dummy_Blocksparse
        # def zero(*args,**kwargs):
        #     matrix = ta.matrix
        #     mat = matrix(0j, complex)
        #     return mat
        # fakelist = np.asarray([(1,1)], dtype=gint_dtype)
        # dummy_Blocksparse = BlockSparseMatrix2(fakelist,fakelist,fakelist, zero)
        # dummy_mat_list = [dummy_Blocksparse]*(self.N_ops)
        # op_mat_list = []*(self.N_ops)
        # op_mat_rev_list = []*(self.N_ops)

        # store necessary BlockSparseMatrices if not unique
        cdef BlockSparseMatrix2 dummy_BSmatrix
        cdef complex ** pointerop_blocks = \
                            <complex **>calloc(self.Nhops_tot, sizeof(complex*))
        cdef complex ** pointerop_rev_blocks = \
                            <complex **>calloc(self.Nhops_tot, sizeof(complex*))
        cdef int numSites, site
        cdef int idx_count = 0
        for i in range(self.N_ops):
            numSites = self.vecauxwhere_list[i+1]-self.vecauxwhere_list[i]
            # if unique_array[i]:
            if self.unique_vector[i]:
                _tmp_mat = self.oplist[i]
                opMat_array[i] = <complex*> &_tmp_mat[0, 0]
                idx_count += numSites
            else:
                if self._bound_operator_list[i]:
                    dummy_BSmatrix = self._bound_operator_list[i]
                else:
                    dummy_BSmatrix = self._eval_operator(i, args, params)

                # store data(=matels) of BlockSparseMat in array
                for site in range(numSites):
                    pointerop_blocks[idx_count] = dummy_BSmatrix.get(site)
                    idx_count += 1


        # op_mat_list = np.asarray(dummy_mat_list)

        # create O_yx (instead of O_xy) if needed
        idx_count = 0
        if self.withRevTerm:
            for i in range(self.N_ops):
                numSites = self.vecauxwhere_list[i+1]-self.vecauxwhere_list[i]
                # if unique_array[i]:
                if self.unique_vector[i]: # transposed needed? probably not!
                    _tmp_mat = self.oplist[i]
                    opMat_rev_array[i] = <complex*> &_tmp_mat[0, 0]
                    idx_count += numSites
                else:
                    if self._bound_operator_rev_list[i]:
                        dummy_BSmatrix = self._bound_operator_rev_list[i]
                    else:
                        dummy_BSmatrix = self._eval_operator(i, args, params, rev=True)

                    # store data(=matels) of BlockSparseMat in array
                    numSites = self.vecauxwhere_list[i+1]-self.vecauxwhere_list[i]
                    for site in range(numSites):
                        pointerop_rev_blocks[idx_count] = dummy_BSmatrix.get(site)
                        idx_count += 1
            # op_mat_rev_list = np.asarray(dummy_mat_list)
        ## COMMENT: For hermitian operators, an additional list would not be
        ## needed, but it would mean making the code more complex.

        ######COMMENT: Could the following be outsourced to not appear here?
        ### In the following, many c-arrays are created. For some of them
        ### the data is already stored in MemoryViews and is here copied to
        ### the c-arrays, which is of course inefficient (but still faster
        ### than using MemVies in the recrusive functions!). Instead, these
        ### pointers should be created already in __init__.
        ### -> TODO: move allocation and copying of most of these objects to __init__!
        ### PROBLEM: How to create pointer as public class variable?
        cdef int * x_norbs = <int*>calloc(self.N_ops+1, sizeof(int))

        # cdef int * pointerauxwhere_list = <int*>calloc(self.N_ops+1, sizeof(int))
        # for i in range(self.N_ops+1):
        #     pointerauxwhere_list[i] = self.vecauxwhere_list[i]
        #
        # cdef int * pointerwherepos_neigh = <int*>calloc(len(self.vecwherepos_neigh), sizeof(int))
        # for i in range(len(self.vecwherepos_neigh)):
        #     pointerwherepos_neigh[i] = self.vecwherepos_neigh[i]
        #
        # cdef int * pointerauxpos_list = <int*>calloc(len(self.vecauxpos_list), sizeof(int))
        # for i in range(len(self.vecauxpos_list)):
        #     pointerauxpos_list[i] = self.vecauxpos_list[i]

        #
        # cdef complex * pointerbra = <complex*>calloc(len(bra), sizeof(complex))
        # for i in range(len(bra)):
        #     pointerbra[i] = bra[i]
        #
        # cdef complex * pointerket = <complex*>calloc(len(ket), sizeof(complex))
        # for i in range(len(ket)):
        #     pointerket[i] = ket[i]

        # cdef complex * pointerout_data = <complex*>calloc(len(out_data), sizeof(complex))
        # for i in range(len(out_data)):
        #     pointerout_data[i] = out_data[i]

        # cdef complex ** pointerop_blocks = \
        #                     <complex **>calloc(self.Nhops_tot, sizeof(complex*))
        # cdef complex ** pointerop_rev_blocks = \
        #                     <complex **>calloc(self.Nhops_tot, sizeof(complex*))
        # cdef int depth
        # cdef int dummy = 0
        # for depth in range(self.N_ops):
        #     i = self.vecauxwhere_list[depth+1]-self.vecauxwhere_list[depth]
        #     for iSite in range(i):
        #         # if not unique_array[depth]:
        #         if not self.unique_vector[depth]:
        #             dummy_BSmatrix = op_mat_list[depth]
        #             pointerop_blocks[dummy] = dummy_BSmatrix.get(iSite)
        #             if self.withRevTerm:
        #                 dummy_BSmatrix = op_mat_rev_list[depth]
        #                 pointerop_rev_blocks[dummy] = dummy_BSmatrix.get(iSite)
        #         dummy += 1

        # get wf starting indices and norbs at sites in where and store in c-array
        cdef int length_lastblock
        length_lastblock = self.Nhops_tot - self.vecauxwhere_list[self.N_ops-1]
        # allocate c array

        cdef int * pointerblockshapes =  \
                                    <int *>calloc(self.Nhops_tot, sizeof(int))
        cdef int * pointerket_start_positions =  \
                                    <int *>calloc(length_lastblock, sizeof(int))
        # only needed if first operator is onsite -- for efficiency only
        cdef int * pointerbra_start_positions
        if self._isonsite[0]:
            pointerbra_start_positions =  \
                        <int *>calloc(self.vecauxwhere_list[1], sizeof(int))
        else:
            pointerbra_start_positions =  \
                        <int *>calloc(1, sizeof(int))

        cdef int norb_dum, wf_st_dum
        for i in range(len(self.where_flat)):
            ### TODO: _get_orbs could be done in __init__ since norbs do not change
            site = self.where_flat[i,1]
            _get_orbs(self._site_ranges, site, &wf_st_dum, &norb_dum)
            pointerblockshapes[i] = norb_dum
            # only last block
            if i > (self.vecauxwhere_list[self.N_ops-1] - 1):
                j =  i - self.vecauxwhere_list[self.N_ops-1]
                pointerket_start_positions[j] = wf_st_dum
            # only first block if first op is onsite to avoid calling _get_orbs again
            if self._isonsite[0] and i < self.vecauxwhere_list[1]:
                pointerbra_start_positions[i] = wf_st_dum
        ### END copying process of MemViews to pointers


        # main loop
        cdef int ket_start = 0
        cdef int bra_start
        cdef int ia  # where given Site is to be found in wherelist
        cdef int a # site ID
        cdef int data_count = 0 #counts the number of output-data calculated

        if op == MAT_ELS:
            # loop over all a-b-hoppings
            for ia in range(self.vecauxwhere_list[1]):
                # get the first operator matrix if necessary (ie not unique)
                # if not unique_array[0]:
                if not self.unique_vector[0]:
                    opMat_array[0] = pointerop_blocks[ia]
                    if self.withRevTerm:
                        opMat_rev_array[0] = pointerop_rev_blocks[ia]

                # get wf start index and number of orbitals
                if self._isonsite[0]:
                    wf_st_dum = pointerbra_start_positions[ia]
                    norb_dum = pointerblockshapes[ia]
                else:
                    a = self.where_flat[ia,0]
                    _get_orbs(self._site_ranges, a, &wf_st_dum, &norb_dum)
                bra_start = wf_st_dum
                x_norbs[0] = norb_dum
                x_norbs[1] = pointerblockshapes[ia]

                # call recursive function to get all needed Mats for given Sites
                sitePath_recFunc(out_data,
                                 &data_count, ia, 1, ket_start, bra_start,
                                 # pointerauxwhere_list,
                                 # pointerwherepos_neigh,
                                 # pointerauxpos_list,
                                 self.vecauxwhere_list,
                                 self.vecwherepos_neigh,
                                 self.vecauxpos_list,
                                 x_norbs,
                                 opMat_array,
                                 opMat_rev_array,
                                 pointerop_blocks,
                                 pointerop_rev_blocks,
                                 pointerblockshapes,
                                 pointerket_start_positions,
                                 self.withRevTerm, self.const_fac,
                                 self.N_ops,
                                 #unique_array,
                                 self.unique_vector,
                                 bra, ket
                                 # pointerbra, pointerket
                                 )
                # END OF LOOP OVER ALL SITE COMBINATIONS

            # # gather data for output
            # assert data_count == self.out_data_length
            # for i in range(self.out_data_length):
            #     out_data[i] = pointerout_data[i]


        elif op == ACT:
            raise NotImplementedError()


        free(x_norbs)
        # free(unique_array)
        free(opMat_array)
        free(opMat_rev_array)
        # free(pointerauxwhere_list)
        # free(pointerwherepos_neigh)
        # free(pointerauxpos_list)
        # free(pointerket)
        # free(pointerbra)
        # free(pointerout_data)
        free(pointerop_blocks)
        free(pointerop_rev_blocks)
        free(pointerblockshapes)
        free(pointerket_start_positions)
        free(pointerbra_start_positions)



    cdef BlockSparseMatrix2 _eval_operator(self, int i, args, params, rev=False):
        """Evaluate the operator matrices on all corresponding elements of `where`"""

        assert not (args and params)
        cdef int start, end
        params = params or {}
        matrix = ta.matrix

        opfunc = self.oplist[i]

        check_hermiticity = self.check_hermiticity

        if self._isonsite[i]:
            required, defaults, takes_kw = self._onsite_params_info[i]
            invalid_params = set(params).intersection(set(defaults))
            if invalid_params:
                raise ValueError("Parameters {} have default values "
                                 "and may not be set with 'params'"
                                 .format(', '.join(invalid_params)))

            if params and not takes_kw:
                params = {pn: params[pn] for pn in required}

            def get_opmat(a, a_norbs, b, b_norbs):
                mat = matrix(opfunc(a, *args, **params), complex)
                _check_operator(mat, opfunc, args, params,
                                a, a_norbs, b, b_norbs, check_hermiticity)
                return mat

         ###XXX: Checks for sanity of 'hop_func' are missing (as opposed to onsite)
        else:
            def get_opmat(a, a_norbs, b, b_norbs):
                mat = matrix(opfunc(a, b, *args, params=params), complex)
                _check_operator(mat, opfunc, args, params,
                                a, a_norbs, b, b_norbs, check_hermiticity)
                return mat
        # use only those parts of self.where_flat, which correspond to the
        # given operator
        start = self.vecauxwhere_list[i]
        end = self.vecauxwhere_list[i+1]
        if not rev:
            auxhops = self.where_flat[start:end]
        else:
            # revert order of hopping
            auxhops = np.asarray([(hop[1],hop[0]) for hop in self.where_flat[start:end]], dtype=gint_dtype)
        offsets, norbs = kwant.operator._get_all_orbs(auxhops, self._site_ranges)
        return BlockSparseMatrix2(auxhops, offsets, norbs, get_opmat)









### Utility functions for the heat current

def _create_where_list_from_added_sites(fsyst, intracell_sites, intercell_sites):
    r"""
    THIS IS AN AUXILIARY FUNCTION ONLY NEEDED FOR THE CASE OF THE HEAT CURRENT IN
    A LEAD, WHICH IS WHY IT SHOULD PROBABLY NOT BE IN THE GENERAL OPERATOR MODULE!

    Creates where list from the sitelists 'intracell_sites' and 'intercell_sites' for (lead) energy current calculation.

    Parameters
    ----------
    intracell_sites: list of all sites in 1st lead unit cell (of type 'int' or
                        instance of kwant.builder.Site)

    intercell_sites: list of all sites in 2nd lead unit cell (of type 'int' or
                        instance of kwant.builder.Site)

    Returns
    -------
    - where: list of lists of hoppings (tupels of 'kwant.builder.Site'):
    [ [all center-intra-hoppings], [all intra-intra- and inter-intra-hoppings]]
    """

    where = [None, None]

    #auxlists to store hoppings
    j_whereaux = []
    central_whereaux = []

    bool_builderSites = False
    # fill neighborlists;
    for intrasite in intracell_sites:
        if isinstance(intrasite, kwant.builder.Site):
            ind_intrasite = fsyst.id_by_site[intrasite]
            bool_builderSites = True
        else:
            ind_intrasite=intrasite

        assert(type(ind_intrasite) == int)

        for iedge in fsyst.graph.out_edge_ids(ind_intrasite):
            neighbor = fsyst.graph.head(iedge)
            if bool_builderSites:
                neighbor = fsyst.sites[neighbor]
            #neighbor in second unit cell of lead:
            if neighbor in set(intercell_sites):
                j_whereaux.append((intrasite, neighbor))
            #neighbor in first unit cell of lead:
            elif neighbor in set(intracell_sites):
                j_whereaux.append((intrasite, neighbor)) ### each hopping atm twice (i.e. in both directions), but that is not a problem, only some redundance
            #neighbor in scattering region
            else:
                central_whereaux.append((neighbor, intrasite))

    where[0] = central_whereaux[:]
    where[1] = j_whereaux[:]

    del central_whereaux, j_whereaux


    return where



def _create_list_of_certain_neighbors(fsyst, initial_list, forbidden_list):
    r"""
    THIS IS AN AUXILIARY FUNCTION ONLY NEEDED FOR THE CASE OF THE HEAT CURRENT IN
    A LEAD, WHICH IS WHY IT SHOULD PROBABLY NOT BE IN THE GENERAL OPERATOR MODULE!

    Creates a list of sites, which are neighbors (connected in the Hamiltonian)
    of the sites in 'initial_list' but which are neither in 'forbidden_list' nor
    in 'initial_list'.
    Used for the shifted energy current in the heat current.

    Parameters
    ----------
    initial_list: list of sites, either as `int` or `~kwant.builder.Site`
    finitial_list: list of sites, either as `int` or `~kwant.builder.Site`

    Returns
    -------
    list of sites as `int`
    """
    #check type of sites in the given lists and convert to int if needed
    if isinstance(initial_list[0], kwant.builder.Site):
        initial_list = list(fsyst.id_by_site[s] for s in initial_list)
    if isinstance(forbidden_list[0], kwant.builder.Site):
        forbidden_list = list(fsyst.id_by_site[s] for s in forbidden_list)
    assert type(initial_list[0]) == int
    assert type(forbidden_list[0]) == int

    # create list in which the neighbors of 'initial_list' are stored which are
    # not in 'forbidden_list' nor in 'initial_list'.
    neighbor_list = []
    for i in initial_list:
        assert(type(i) == int)
        for iedge in fsyst.graph.out_edge_ids(i):
            neighbor = fsyst.graph.head(iedge)
            #neighbor in forbidden_list -> do nothing:
            if neighbor in set(forbidden_list):
                pass
            #neighbor in initial_list -> do nothing:
            elif neighbor in set(initial_list):
                pass
            #neighbor already in neighbor_list -> do nothing:
            elif neighbor in set(neighbor_list):
                pass
            #neighbor not yey in neighbor_list -> add it
            else:
                neighbor_list.append(neighbor)

    return neighbor_list















# 100% copy from kwant.operator
cdef class BlockSparseMatrix2:
    """A sparse matrix stored as dense blocks.

    Parameters
    ----------
    where : gint[:, :]
        ``Nx2`` matrix or ``Nx1`` matrix: the arguments ``a``
        and ``b`` to be used when evaluating ``f``. If an
        ``Nx1`` matrix, then ``b=a``.
    block_offsets : gint[:, :]
        The row and column offsets for the start of each block
        in the sparse matrix: ``(row_offset, col_offset)``.
    block_shapes : gint[:, :]
        ``Nx2`` array: the shapes of each block, ``(n_rows, n_cols)``.
    f : callable
        evaluates matrix blocks. Has signature ``(a, n_rows, b, n_cols)``
        where all the arguments are integers and
        ``a`` and ``b`` are the contents of ``where``. This function
        must return a matrix of shape ``(n_rows, n_cols)``.

    Attributes
    ----------
    block_offsets : gint[:, :]
        The row and column offsets for the start of each block
        in the sparse matrix: ``(row_offset, col_offset)``.
    block_shapes : gint[:, :]
        The shape of each block: ``(n_rows, n_cols)``
    data_offsets : gint[:]
        The offsets of the start of each matrix block in `data`.
    data : complex[:]
        The matrix of each block, stored in row-major (C) order.
    """

    cdef public int[:, :] block_offsets, block_shapes
    cdef public int[:] data_offsets
    cdef public complex[:] data

    @cython.embedsignature
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, int[:, :] where, int[:, :] block_offsets,
                  int[:, :] block_shapes, f):
        if (block_offsets.shape[0] != where.shape[0] or
            block_shapes.shape[0] != where.shape[0]):
            raise ValueError('Arrays should be the same length along '
                             'the first axis.')
        self.block_shapes = block_shapes
        self.block_offsets = block_offsets
        self.data_offsets = np.empty(where.shape[0], dtype=gint_dtype)
        ### calculate shapes and data_offsets
        cdef int w, data_size = 0
        for w in range(where.shape[0]):
            self.data_offsets[w] = data_size
            data_size += block_shapes[w, 0] * block_shapes[w, 1]
        ### Populate data array
        self.data = np.empty((data_size,), dtype=complex)
        cdef complex[:, :] mat
        cdef int i, j, off, a, b, a_norbs, b_norbs
        for w in range(where.shape[0]):
            off = self.data_offsets[w]
            a_norbs = self.block_shapes[w, 0]
            b_norbs = self.block_shapes[w, 1]
            a = where[w, 0]
            b = a if where.shape[1] == 1 else where[w, 1]
            # call the function that gives the matrix
            mat = f(a, a_norbs, b, b_norbs)
            # Copy data
            for i in range(a_norbs):
                for j in range(b_norbs):
                    self.data[off + i * b_norbs + j] = mat[i, j]

    cdef complex* get(self, int block_idx):
        return  <complex*> &self.data[0] + self.data_offsets[block_idx]

    def __getstate__(self):
        return tuple(map(np.asarray, (
            self.block_offsets,
            self.block_shapes,
            self.data_offsets,
            self.data
        )))

    def __setstate__(self, state):
        (self.block_offsets,
         self.block_shapes,
         self.data_offsets,
         self.data,
        ) = state


#99% copy from kwant.operator._check_ham -> int instead of gint
cdef int _check_operator(complex[:, :] H, func, args, params,
                    int a, int a_norbs, int b, int b_norbs,
                    int check_hermiticity) except -1:
    "Check operator matrix for correct shape and hermiticity."
    if H.shape[0] != a_norbs and H.shape[1] != b_norbs:
        raise UserCodeError(kwant.operator._shape_msg.format('Operator'))
    if check_hermiticity:
        # call the "partner" element if we are not on the diagonal
        H_conj = H if a == b else ta.matrix(func(b, a, *args, params=params),
                                                complex)
        if not _is_herm_conj2(H_conj, H):
            raise ValueError(kwant.operator._herm_msg.format('Operator'))
    return 0

# #99% copy from kwant.operator -> int instead of gint
# cdef int _check_ham2(complex[:, :] H, ham, args, params,
#                     int a, int a_norbs, int b, int b_norbs,
#                     int check_hermiticity) except -1:
#     "Check Hamiltonian matrix for correct shape and hermiticity."
#     if H.shape[0] != a_norbs and H.shape[1] != b_norbs:
#         raise UserCodeError(kwant.operator._shape_msg.format('Hamiltonian'))
#     if check_hermiticity:
#         # call the "partner" element if we are not on the diagonal
#         H_conj = H if a == b else ta.matrix(ham(b, a, *args, params=params),
#                                                 complex)
#         if not _is_herm_conj2(H_conj, H):
#             raise ValueError(kwant.operator._herm_msg.format('Hamiltonian'))
#     return 0

#100% copy of kwant.operator
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _is_herm_conj2(complex[:, :] a, complex[:, :] b,
                       double atol=1e-300, double rtol=1e-13) except -1:
    "Return True if `a` is the Hermitian conjugate of `b`."
    assert a.shape[0] == b.shape[1]
    assert a.shape[1] == b.shape[0]

    # compute max(a)
    cdef double tmp, max_a = 0
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tmp = a[i, j].real * a[i, j].real + a[i, j].imag * a[i, j].imag
            if tmp > max_a:
                max_a = tmp
    max_a = math.sqrt(max_a)

    cdef double tol = rtol * max_a + atol
    cdef complex ctmp
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ctmp = a[i, j] - b[j, i].conjugate()
            tmp = ctmp.real * ctmp.real + ctmp.imag * ctmp.imag
            if tmp > tol:
                return False
    return True




#100% copy from kwant.operator
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _get_orbs(int[:, :] site_ranges, int site,
                    int *start_orb, int *norbs):
    """Return the first orbital of this site and the number of orbitals"""
    cdef int run_idx, first_site, norb, orb_offset, orb
    # Calculate the index of the range that contains the site.
    run_idx = _bisect(site_ranges[:, 0], site) - 1
    first_site = site_ranges[run_idx, 0]
    norb = site_ranges[run_idx, 1]
    orb_offset = site_ranges[run_idx, 2]
    # calculate the slice
    start_orb[0] = orb_offset + (site - first_site) * norb
    norbs[0] = norb




#99% copy from kwant.operator -> int instead of gint
def _get_tot_norbs(syst):
    cdef int _unused, tot_norbs
    is_infinite_system = isinstance(syst, InfiniteSystem)
    n_sites = syst.cell_size if is_infinite_system else syst.graph.num_nodes
    _get_orbs(np.asarray(syst.site_ranges, dtype=gint_dtype),
              n_sites, &tot_norbs, &_unused)
    return tot_norbs


#99% copy from kwant.operator -> int instead of gint
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _bisect(int[:] a, int x):
    "bisect.bisect specialized for searching `site_ranges`"
    cdef int mid, lo = 0, hi = a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

_shape_msg = ('{0} matrix dimensions do not match '
              'the declared number of orbitals')

_herm_msg = ('{0} matrix is not hermitian, use the option '
             '`check_hermiticity=True` if this is intentional.')

#  #99% copy from kwant.operator -> int instead of gint
# cdef int _check_onsite2(complex[:, :] M, int norbs,
#                        int check_hermiticity) except -1:
#     "Check onsite matrix for correct shape and hermiticity."
#     if M.shape[0] != M.shape[1]:
#         raise UserCodeError('Onsite matrix is not square')
#     if M.shape[0] != norbs:
#         raise UserCodeError(_shape_msg.format('Onsite'))
#     if check_hermiticity and not _is_herm_conj2(M, M):
#         raise ValueError(_herm_msg.format('Onsite'))
#     return 0
