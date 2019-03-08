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
cimport kwant.operator
#from kwant.graph.defs import gint_dtype
from kwant.system import InfiniteSystem
###FOR _check_ham
from kwant._common import UserCodeError, get_parameters

from libc.stdlib cimport free, calloc, malloc
from cython.view cimport array as cvarray
from inspect import signature

import numpy as np

from kwant.graph.defs cimport gint
# from kwant cimport operator
from kwant.graph.defs import gint_dtype

# gint_dtype = np.int32
# cdef extern from "defs.h":
#     ctypedef signed int gint

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
    r"""An operator for calculating general ???.

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


cdef class offEnergyCurrent:
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
    def __init__(self, syst, where, *, check_hermiticity=True, sum=False, relPathList=None):

        self.hamil1 = Operator(syst, syst.hamiltonian, where[0], check_hermiticity=check_hermiticity, sum=sum)
        self.hamil2 = Operator(syst, syst.hamiltonian, where[1], check_hermiticity=check_hermiticity, sum=sum)

        self.offEn = Op_Product(self.hamil1, self.hamil2, withRevTerm=-1, const_fac=1j, check_hermiticity=check_hermiticity, sum=sum, in_relPathList=relPathList)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return self.offEn.bind(ops_tobebound='all', args=args, params=params)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        return self.offEn(bra, ket, args, params=params)

    def get_sitechains(self):
        return [list(path) for path in self.offEn.sitechains]



#recursive function to get all combinations of sites of the sums
cdef pathlist_recfunc(object path_list_data, int[:] path_list_aux, int iSite,
                        int depth,
                        int[:] auxwhere_list,
                        int[:] auxpos_list,
                        int[:] wherepos_neigh,
                        int N_ops):
    """
    Recursive function to find all possible paths.
    This recursive function is the heart of the method
    `_create_sitechains_output(..)`.
    """
    # add the next site to the chain
    cdef int offset = auxwhere_list[depth-1]
    path_list_aux[depth] = iSite

    cdef int i, auxpos_list_index, startindex_neighbors, num_neighbors, iNextSite
    #recursion end: copy sitechain_aux to sitechain_data
    if depth == N_ops:
        #sliced for copy (avoid reference)
        path_list_data.append(path_list_aux[:])
    #recursion step: find neighbor and call the next step
    else:
        assert depth < (N_ops)
        auxpos_list_index = offset + iSite
        startindex_neighbors = auxpos_list[auxpos_list_index]
        num_neighbors = auxpos_list[auxpos_list_index+1] - startindex_neighbors
        for i in range(num_neighbors):
            iNextSite = wherepos_neigh[startindex_neighbors + i]
            pathlist_recfunc(path_list_data, path_list_aux, iNextSite, depth+1,
                               auxwhere_list,
                               auxpos_list,
                               wherepos_neigh,
                               N_ops)
    # end recursive function


def _create_path_list(auxwhere_list, auxpos_list, wherepos_neigh, N_ops):
    r"""
    Returns a list, which has the same structure as the Operator
    output_data and which can therefore be used to link the output to a path
    of sites.
    The structure of this function is very similar to sitePath_recFunc with the
    difference that here, only the site-chains are stored and that it is called
    only in the `Op_Product.__init__()` instead of `Operator._operate()`.
    """

    cdef int ia

    path_list_data = []
    cdef int[:] path_list_aux
    path_list_aux = np.zeros(N_ops, dtype=gint_dtype)
    # loop over all 'a'-Sites and initialize first step (for 'b'-Sites)
    for ia in range(auxwhere_list[1]):
        # call recursive function
        pathlist_recfunc(path_list_data, path_list_aux, ia, 1,
                         auxwhere_list, auxpos_list, wherepos_neigh, N_ops)

    if path_list_data == []:
        raise UserCodeError('No path along the given where\'s of the different operators is found! Calculation is aborted!')
    return np.asarray(path_list_data, dtype=gint_dtype)





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
        cdef gint[:,:] where_fakehops
        cdef gint i, j
        if self.willNotBeCalled:
            where_fakehops = None
        else:
            _where = kwant.operator._normalize_site_where(syst, where)
            _where = _where.reshape(-1)
            where_fakehops = np.empty((len(_where),2), dtype=gint_dtype)
            where_fakehops_transposed = np.empty((2,len(_where)), dtype=gint_dtype)
            where_fakehops_transposed[0] = _where
            where_fakehops_transposed[1] = _where
            where_fakehops = np.transpose(where_fakehops_transposed)
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
    const_fac: a constant complex factor which is multiplied to the result of the operator
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
    in_relPathList : list of lists
        If the paths, i.e. which hoppings of the n-th operator are to be
        connected with which hoppings of the (n+1)-th operator, are already
        known they can be given by this list of lists. Each list=path is
        represented by the relative position of a hopping in the where of the
        corresponding Hamiltonian.



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

    Unfortunately, variable names do not follow a strict convention. Some of the
    names are not precise or even misleading. (To be fixed.)
    """

    def __init__(self, *ops, withRevTerm=0, const_fac=1, check_hermiticity=True,
                 sum=False, willNotBeCalled=False, in_relPathList=None):

        assert len(ops) > 1
        cdef int i,j,k
        # assert that system is the same for all operators
        for i in range(1, len(ops)):
            assert ops[0].syst == ops[i].syst

        # store class variables
        self.syst = ops[0].syst
        self._site_ranges = ops[0]._site_ranges
        self.check_hermiticity = check_hermiticity
        self.sum = sum
        self.withRevTerm = withRevTerm
        self.const_fac = const_fac

        self.max_orbs = 0
        for op in ops:
            if op.max_orbs > self.max_orbs:
                self.max_orbs = op.max_orbs

        self.mult_vec0 = np.empty(self.max_orbs, dtype=complex)
        self.mult_vec1 = np.empty(self.max_orbs, dtype=complex)
        self.tmp_vec0 = np.empty(self.max_orbs, dtype=complex)
        self.tmp_vec1 = np.empty(self.max_orbs, dtype=complex)


        # total amount of operators (given ops could be already products of operators)
        self.N_ops = 0
        for op in ops:
            self.N_ops += op.N_ops

        # copy data from ops to product operator
        self.oplist = [None] * self.N_ops
        self._onsite_params_info = [None] * self.N_ops
        self._isonsite = np.zeros(self.N_ops, dtype=gint_dtype)
        self.unique_list = np.zeros(self.N_ops, dtype=gint_dtype)
        self._bound_operator_list = [None] * self.N_ops
        self._bound_operator_rev_list = [None] * self.N_ops
        cdef int op_idx = 0
        for i, op in enumerate(ops):
            # any operator in ops can already be a product of many operators
            for j in range(len(op.oplist)):
                assert len(op.oplist) == len(op._onsite_params_info)
                assert len(op.oplist) == len(op.unique_list)
                assert len(op.oplist) == len(op._isonsite)
                assert len(op.oplist) == len(op._bound_operator_list)
                assert len(op.oplist) == len(op._bound_operator_rev_list)
                self.oplist[op_idx] = op.oplist[j]
                self._onsite_params_info[op_idx] = op._onsite_params_info[j]
                self._isonsite[op_idx] = op._isonsite[j]
                self.unique_list[op_idx] = op.unique_list[j]
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
            assert not in_relPathList
            # we already have all that we need if this operator is not to be called
            return

        ### product-operator can be called -> we need:
        ### where_flat, auxlists, sitechains, output_length and dict_lists, rel_path_list


        cdef int lstart, lend, rstart, rend, hop_el, hop
        cdef int dict_idx = 0

        cdef int start = 0
        cdef int end = 0
        cdef int aux_idx = 1
        cdef int offset = 0
        cdef int num_added_fakehops
        cdef int auxpos_list_len, auxpos_value_offset, auxpos_idx
        self.auxwhere_list = np.zeros(self.N_ops+1, dtype=gint_dtype)
        cdef gint[:,:] dummylist


        # initial operators not to be called but product is to be called
        if ops[0].willNotBeCalled and ops[1].willNotBeCalled:
            print("Initialize product of uncallable operators. The product is callable.")
            assert not in_relPathList
            _allSites = kwant.operator._normalize_site_where(self.syst, None)
            # atm, even for onsite operators, the elements of `where` are
            #(fake)hoppings (from one site to the same site)
            _fakehops = [(site[0],site[0]) for site in _allSites]
            self.where_flat = np.asarray(_fakehops * self.N_ops, dtype=gint_dtype)
            self.Nhops_tot = self.N_ops*len(_fakehops)


            ### The following lists are auxiliary lists to find the correct paths.
            ### In this case, they are trivial and no path finding algorithm has
            ### to be used.

            # self.auxwhere_list
            self.auxwhere_list = np.asarray(
                        range(0,self.Nhops_tot+1,len(_fakehops)),
                        dtype=gint_dtype)
            # self.wherepos_neigh
            self.wherepos_neigh = np.asarray(list(range(len(_fakehops)))*(self.N_ops-1), dtype=gint_dtype)
            # self.auxpos_list
            self.auxpos_list = np.asarray(list(range(0,len(_fakehops)*(self.N_ops-1)+1)), dtype=gint_dtype)
            # self.sitechains
            self.sitechains = np.asarray([range(0,len(_fakehops))], dtype=gint_dtype)
            # self.rel_path_list
            self.rel_path_list = np.asarray([([i]*self.N_ops) for i in range(0,len(_fakehops)) ], dtype=gint_dtype)

        # one initial operator not to be called, the other one is callable
        elif ops[0].willNotBeCalled != ops[1].willNotBeCalled:
            #TODO: have less in distinguishing if-else statements
            print("Initialize product of one callable operator and one uncallable operator.")
            assert not in_relPathList
            if ops[0].willNotBeCalled:
                opCallable = ops[1]
                opNotCallable = ops[0]
                # information for where_flat, which is generated after this `if`
                hop_el = 0
                start = 0
                end = opCallable.auxwhere_list[1]

                offset_uncallable = 0
                offset_callable = opNotCallable.N_ops * (end-start)
                num_added_fakehops = (end-start) * opNotCallable.N_ops


                ### no path finding algorithm has to be applied in this case
                ### since one of the operators is `willNotBeCalled`
                # self.whereaux_list
                for i in range(opNotCallable.N_ops):
                    self.auxwhere_list[i] = (end-start) * i
                for i in range(opCallable.N_ops+1):
                    j = i+opNotCallable.N_ops
                    self.auxwhere_list[j] = \
                     opCallable.auxwhere_list[i] + num_added_fakehops

                # rel_path_list
                self.rel_path_list = np.zeros((len(opCallable.rel_path_list),self.N_ops), dtype=gint_dtype)
                dummylist = opCallable.rel_path_list
                for i in range(len(opCallable.rel_path_list)):
                    #copy the paths so far
                    for j in range(opCallable.N_ops):
                        k = j+opNotCallable.N_ops
                        self.rel_path_list[i,k] = dummylist[i,j]
                    # extend path for onsite, i.e. repeat the first site of callable op
                    for j in range(opNotCallable.N_ops):
                        self.rel_path_list[i,j] = dummylist[i,0]


                # self.wherepos_neigh
                self.wherepos_neigh = np.asarray(
                         list(range(end-start))*opNotCallable.N_ops+list(opCallable.wherepos_neigh),
                         dtype=gint_dtype)

                # self.auxpos_list
                if len(opCallable.auxpos_list) == 0:
                    auxpos_list_len = num_added_fakehops + 1  #last element is length of wherepos_neigh
                else:
                    auxpos_list_len = num_added_fakehops + len(opCallable.auxpos_list)
                self.auxpos_list = np.zeros(auxpos_list_len, dtype=gint_dtype)

                #add additional absolute 'neigh'-positions in wherepos_neigh
                for i in range(num_added_fakehops):
                    self.auxpos_list[i] = i
                # copy old data
                for i in range(len(opCallable.auxpos_list)):
                    j = i+num_added_fakehops
                    self.auxpos_list[j] = \
                                 opCallable.auxpos_list[i] + num_added_fakehops
                if len(opCallable.auxpos_list) == 0:
                    self.auxpos_list[num_added_fakehops] = num_added_fakehops

            elif ops[1].willNotBeCalled:
                opCallable = ops[0]
                opNotCallable = ops[1]

                # information for where_flat, which is generated after this `if`
                hop_el = 1
                start = opCallable.auxwhere_list[-2]
                end = opCallable.auxwhere_list[-1]

                offset_uncallable = opCallable.Nhops_tot
                offset_callable = 0
                num_added_fakehops = (end-start) * opNotCallable.N_ops

                ### no path finding algorithm has to be applied in this case
                ### since one of the operators is `willNotBeCalled`
                # self.whereaux_list
                for i in range(opCallable.N_ops):
                    self.auxwhere_list[i] = opCallable.auxwhere_list[i]
                for i in range(opNotCallable.N_ops+1):
                    j = i+opCallable.N_ops
                    self.auxwhere_list[j] = \
                      (end-start) * i + opCallable.auxwhere_list[-1]

                # self.wherepos_neigh
                self.wherepos_neigh = np.asarray(
                         list(opCallable.wherepos_neigh)+
                         list(range(end-start))*(opNotCallable.N_ops),
                         dtype=gint_dtype)

                # rel_path_list
                self.rel_path_list = np.zeros((len(opCallable.rel_path_list),self.N_ops), dtype=gint_dtype)
                dummylist = opCallable.rel_path_list
                for i in range(len(opCallable.rel_path_list)):
                    #copy the paths so far
                    for j in range(opCallable.N_ops):
                        self.rel_path_list[i,j] = dummylist[i,j]
                    # extend path for onsite, i.e. repeat the first site of callable op
                    for j in range(opNotCallable.N_ops):
                        k = j+opCallable.N_ops
                        self.rel_path_list[i,k] = dummylist[i,-1]

                # self.auxpos_list
                if len(opCallable.auxpos_list) == 0:
                    auxpos_list_len = num_added_fakehops + 1  #last element is length of wherepos_neigh
                else:
                    auxpos_list_len = num_added_fakehops + len(opCallable.auxpos_list)
                self.auxpos_list = np.zeros(auxpos_list_len, dtype=gint_dtype)
                # copy old data
                for i in range(len(opCallable.auxpos_list)):
                    self.auxpos_list[i] = opCallable.auxpos_list[i]
                # check for cases in which 'opCallable.auxpos_list' does not exist
                if len(opCallable.auxpos_list) == 0:
                    offset_idx = 0
                    offset_value = 0
                else:
                    offset_idx = len(opCallable.auxpos_list)
                    offset_value = len(opCallable.wherepos_neigh) + 1
                #add additional absolute 'neigh'-positions in wherepos_neigh
                for i in range(num_added_fakehops):
                    j = i+offset_idx
                    self.auxpos_list[j] = i + offset_value
                if len(opCallable.auxpos_list) == 0:
                    self.auxpos_list[num_added_fakehops] = num_added_fakehops

            else:
                raise ValueError("The wrong operator multiplication option was chosen by the code.")

            N_onsites = opNotCallable.N_ops
            self.Nhops_tot = opCallable.Nhops_tot + num_added_fakehops

            # self.where_flat
            self.where_flat = np.zeros((self.Nhops_tot,2), dtype=gint_dtype)
                # add new fakehoppings
            dummylist = opCallable.where_flat
            for i in range( num_added_fakehops ):
                for j in range(2):
                    k = offset_uncallable+i
                    self.where_flat[k,j] = dummylist[start+i, hop_el]
                # copy old hoppings
            for i in range(opCallable.Nhops_tot):
                for j in range(2):
                    k = i+offset_callable
                    self.where_flat[k,j] = dummylist[i,j]

            # self.sitechains
            self.sitechains = opCallable.sitechains



        elif not ops[0].willNotBeCalled and not ops[1].willNotBeCalled:
            print("Initialize product of callable operators.")
            self.Nhops_tot = 0
            for op in ops:
                self.Nhops_tot += op.Nhops_tot
            # self.where_flat and self.auxwhere_list
            self.where_flat = np.zeros((self.Nhops_tot,2), dtype=gint_dtype)
            for op in ops:
                end += op.Nhops_tot
                for i in range(end-start):
                    for j in range(2):
                        self.where_flat[start+i][j] = op.where_flat[i][j]
                for where_offset in op.auxwhere_list[1:]:
                    self.auxwhere_list[aux_idx] = where_offset + start
                    aux_idx += 1
                start = end

            # self.wherepos_neigh and self.auxpos_list
            _list_wherepos_neigh = []
            self.auxpos_list = np.zeros(self.auxwhere_list[-2]+1, dtype = gint_dtype)
            auxpos_value_offset = 0
            auxpos_idx = 0
            for i in range(len(ops)-1):
                #copy old auxlists of i-th operator
                _list_wherepos_neigh = _list_wherepos_neigh + list(ops[i].wherepos_neigh)
                for j, value in enumerate(ops[i].auxpos_list[1:]):
                    auxpos_idx += 1
                    self.auxpos_list[auxpos_idx] = value + auxpos_value_offset

                auxpos_value_offset += len(ops[i].wherepos_neigh)

                # create new dict as tool for creating the new auxlists,
                # wherepos_neigh and auxpos_list, at the product interface
                lstart = ops[i].auxwhere_list[-2]
                lend = ops[i].auxwhere_list[-1]
                rstart = ops[i+1].auxwhere_list[0]  # == 0
                rend = ops[i+1].auxwhere_list[1]
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
                    self.auxpos_list[auxpos_idx] = value + auxpos_value_offset

                auxpos_value_offset += len(_new_wherepos_neigh)

            # copy of old auxlists of last factor
            _list_wherepos_neigh = _list_wherepos_neigh + list(ops[-1].wherepos_neigh)
            for j, value in enumerate(ops[-1].auxpos_list[1:]):
                auxpos_idx += 1
                self.auxpos_list[auxpos_idx] = value + auxpos_value_offset

            self.wherepos_neigh = np.asarray(_list_wherepos_neigh, dtype=gint_dtype)

            # self.sitechains
            if in_relPathList:
                self.rel_path_list = np.asarray(in_relPathList, dtype=gint_dtype)
            else:
                self.rel_path_list = _create_path_list(self.auxwhere_list,
                                                       self.auxpos_list,
                                                       self.wherepos_neigh,
                                                       self.N_ops)

        else:
            raise ValueError("Cannot find appropriate option for the operator multiplication.")

        # self.out_data_length is in all cases the same if product is callable
        self.out_data_length = len(self.rel_path_list)

        # prev/next nonUnique Operator
        self.prev_nonUnique = np.zeros(self.N_ops, dtype=gint_dtype)
        self.next_nonUnique = np.zeros(self.N_ops, dtype=gint_dtype)
        cdef int prev_nonUnique_Op = -1
        cdef int next_nonUnique_Op = -1
        for i in range(self.N_ops):
            if not self.unique_list[i]:
                prev_nonUnique_Op = i
            self.prev_nonUnique[i] = prev_nonUnique_Op
            if not self.unique_list[self.N_ops-1-i]:
                next_nonUnique_Op = self.N_ops-1-i
            self.next_nonUnique[i] = next_nonUnique_Op



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
    cdef public int[:, :] sitechains, rel_path_list
    # flattened where list of hoppings
    cdef public int[:, :]  where_flat
    # 1d-auxlists, mostly for products of operators by helping in path finding
    cdef public int[:] auxwhere_list, wherepos_neigh, auxpos_list
    # in case of bound operators
    cdef public object _bound_operator_list, _bound_operator_rev_list, _onsite_params_info
    # in case of unique operators
    cdef public gint[:] unique_list, _isonsite, prev_nonUnique, next_nonUnique
    # site information for _get_orbs
    cdef public gint[:, :] _site_ranges
    # for the actual calculation
    cdef public int max_orbs
    cdef public complex[:] mult_vec0, mult_vec1, tmp_vec0, tmp_vec1
    # for onsite operators to avoid unnecessary path-finding
    cdef public int willNotBeCalled


    @cython.embedsignature
    def __init__(self, syst, opfunc, in_where, withRevTerm=0, const_fac=1, *, check_hermiticity=False, sum=False):
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
        self.unique_list = np.asarray([not callable(opfunc)],dtype=gint_dtype)
        self.prev_nonUnique = np.asarray([-1],dtype=gint_dtype)
        self.next_nonUnique = np.asarray([-1],dtype=gint_dtype)
        self._site_ranges = np.asarray(syst.site_ranges, dtype=gint_dtype)
        self.max_orbs = np.amax(np.transpose(self._site_ranges)[1])
        # vectors for matrix prooducts (over orbitals) -- initialize only once
        self.mult_vec0 = np.empty(self.max_orbs, dtype=complex)
        self.mult_vec1 = np.empty(self.max_orbs, dtype=complex)
        self.tmp_vec0 = np.empty(self.max_orbs, dtype=complex)
        self.tmp_vec1 = np.empty(self.max_orbs, dtype=complex)

        self._bound_operator_list = [None]
        self._bound_operator_rev_list = [None]


        cdef int i
        # where and its auxlists are not needed if this operator is not being called
        # (can only be the case for onsite operators)
        if self.willNotBeCalled:
            self.where_flat = None
            self.Nhops_tot = 0
            return
        elif self._isonsite[0]:
            self.where_flat = in_where
        else:
            # check if where site are already integers instead of kwant.Sites, because
            # kwant.operator._normalize_hopping_where throws an error in that case
            # if not callable(in_where) and not in_where == None and
            try:
                if type(in_where[0][0]) in [gint_dtype, int]:
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

        self.rel_path_list = np.empty((self.Nhops_tot,1), dtype=gint_dtype)
        for i in range(len(self.where_flat)):
            self.rel_path_list[i,0] = i

        # The following are interesting objects only in case of products of ops.
        # They are trivial in this __init__ method but still needed for the
        # calculation.
        self.out_data_length = self.Nhops_tot
        self.sitechains = self.where_flat
        self.auxwhere_list =  np.asarray((0, self.Nhops_tot), dtype=gint_dtype)

        self.wherepos_neigh = np.asarray([], dtype=gint_dtype)
        self.auxpos_list = np.asarray([0], dtype=gint_dtype)



    @cython.embedsignature
    def get_sitechains(self):
        r"""
        Returns list of all possible paths which are in the same order as
        the output_data-list.
        """
        if self.sitechains.size == 0:
            return [list(path) for path in self.sitechains]
        else:
            ret_list= []
            for path in self.rel_path_list:
                for i, rel_pos in enumerate(path):
                    abs_pos = rel_pos + self.auxwhere_list[i]
                    ret_list.append(self.whereflat[abs_pos,0])
            return ret_list

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
        q.rel_path_list = self.rel_path_list
        q.where_flat = self.where_flat
        q.auxwhere_list = self.auxwhere_list
        q.wherepos_neigh = self.wherepos_neigh
        q.auxpos_list = self.auxpos_list
        q._bound_operator_list = self._bound_operator_list
        q._bound_operator_rev_list = self._bound_operator_rev_list
        q._onsite_params_info = self._onsite_params_info
        q.unique_list = self.unique_list
        q.prev_nonUnique = self.prev_nonUnique
        q.next_nonUnique = self.next_nonUnique
        q._isonsite = self._isonsite
        q._site_ranges = self._site_ranges

        q.max_orbs = self.max_orbs
        q.mult_vec0 = self.mult_vec0
        q.mult_vec1 = self.mult_vec1
        q.tmp_vec0 = self.tmp_vec0
        q.tmp_vec1 = self.tmp_vec1

        q.willNotBeCalled = self.willNotBeCalled

        if ops_tobebound == 'all':
            ops_tobebound = list(range(self.N_ops))
        for index in ops_tobebound:
            if callable(self.oplist[index]):
                q._bound_operator_list[index] = self._eval_operator(index, args, params)
                if not self.check_hermiticity:
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
        cdef complex **opMat_array = \
                          <complex **>calloc(self.N_ops, sizeof(complex*))
        # for reversed order of operators
        cdef complex **opMat_rev_array = \
                        <complex **>calloc(self.N_ops, sizeof(complex*))

        cdef BlockSparseMatrix2 dummy_BSmatrix
        cdef BlockSparseMatrix2[:] op_mat_list
        cdef BlockSparseMatrix2[:] op_mat_rev_list
        # initializing MemView with a dummy BlockSparseMatrix
        cdef BlockSparseMatrix2 dummy_Blocksparse
        def zero(*args,**kwargs):
            matrix = ta.matrix
            mat = matrix(0j, complex)
            return mat
        fakelist = np.asarray([(1,1)], dtype=gint_dtype)
        dummy_Blocksparse = BlockSparseMatrix2(fakelist,fakelist,fakelist, zero)
        dummy_mat_list = [dummy_Blocksparse]*(self.N_ops)

        # store necessary BlockSparseMatrices if not unique
        for i in range(self.N_ops):
            if self.unique_list[i]:
                _tmp_mat = self.oplist[i]
                opMat_array[i] = <complex*> &_tmp_mat[0, 0]
                # op_mat_list[i] = None
            elif self._bound_operator_list[i]:
                dummy_BSmatrix = self._bound_operator_list[i]
                dummy_mat_list[i] = dummy_BSmatrix
            else:
                dummy_BSmatrix = self._eval_operator(i, args, params)
                dummy_mat_list[i] = dummy_BSmatrix

        op_mat_list = np.asarray(dummy_mat_list)

        # create O_yx (instead of O_xy) if needed
        if self.withRevTerm:
            if self.check_hermiticity:
                opMat_rev_array = opMat_array
                op_mat_rev_list = op_mat_list
            else:
                for i in range(self.N_ops):
                    if self.unique_list[i]:  # transposed needed? probably not!
                        _tmp_mat = self.oplist[i]
                        opMat_rev_array[i] = <complex*> &_tmp_mat[0, 0]
                    elif self._isonsite[i]:
                        pass  # dummy_mat_list is the same as in the nonrev case
                    elif self._bound_operator_rev_list[i]:
                        dummy_BSmatrix = self._bound_operator_rev_list[i]
                        dummy_mat_list[i] = dummy_BSmatrix
                    else:
                        dummy_BSmatrix = self._eval_operator(i, args, params, rev=True)
                        dummy_mat_list[i] = dummy_BSmatrix
                op_mat_rev_list = np.asarray(dummy_mat_list)

        # for the matrix products
        cdef gint ket_start, bra_start
        cdef gint o_j, o_i, o_prev, norbs, norbs_next, ipath
        cdef gint nops
        cdef complex tmp_prod
        cdef complex tmp_sum = 0
        cdef complex result
        cdef gint[:] o_prev_list
        o_prev_list = np.zeros(self.N_ops, dtype=gint_dtype)

        cdef int everyOp_unique
        cdef int ref_Op_pos
        cdef int[:] x_norbs
        x_norbs = np.zeros(self.N_ops+1, dtype=gint_dtype)
        cdef int[:] path
        cdef gint site_dummy, a_norbs, a


        if op == MAT_ELS:
            everyOp_unique = (np.sum(self.unique_list) == self.N_ops)

            # MAIN LOOP
            for ipath in range(len(self.rel_path_list)):
                path = self.rel_path_list[ipath]
                # fill needed lists for given path: xnorbs, opMatArray, ket_start, bra_start
                if everyOp_unique:
                    a = self.where_flat[path[0],0]
                    _get_orbs2(self._site_ranges, a, &ket_start, &a_norbs)
                    # only onsite, i.e. everything is the same for every operator
                    bra_start = ket_start
                    x_norbs[0] = a_norbs
                    for nops in range(self.N_ops):
                        x_norbs[nops+1] = a_norbs
                else:
                    # get ket_start
                    if not self.unique_list[self.N_ops-1]:
                        dummy_BSmatrix = op_mat_list[self.N_ops-1]
                        ket_start = dummy_BSmatrix.block_offsets[path[self.N_ops-1],1]
                    else:
                        ref_Op_pos = self.prev_nonUnique[self.N_ops-1]
                        dummy_BSmatrix = op_mat_list[ref_Op_pos]
                        ket_start = dummy_BSmatrix.block_offsets[path[self.N_ops-1],1]

                    # get bra_start, x_norbs[0]
                    if not self.unique_list[0]:
                        dummy_BSmatrix = op_mat_list[0]
                        bra_start = dummy_BSmatrix.block_offsets[path[0],0]
                        x_norbs[0] = dummy_BSmatrix.block_shapes[path[0],0]
                    else:
                        ref_Op_pos = self.next_nonUnique[0]
                        dummy_BSmatrix = op_mat_list[ref_Op_pos]
                        bra_start = dummy_BSmatrix.block_offsets[path[0],0]
                        x_norbs[0] = dummy_BSmatrix.block_shapes[path[0],0]

                    # get x_norbs[nops]
                    for nops in range(self.N_ops):
                        if not self.unique_list[nops]:
                            x_norbs[nops+1] = op_mat_list[nops].block_shapes[path[nops],1]
                        else:
                            if self.prev_nonUnique[nops] != -1:
                                ref_Op_pos = self.prev_nonUnique[nops]
                                x_norbs[nops+1] = op_mat_list[ref_Op_pos].block_shapes[path[nops],1]
                            elif self.next_nonUnique[nops] != -1:
                                ref_Op_pos = self.next_nonUnique[nops]
                                x_norbs[nops+1] = op_mat_list[ref_Op_pos].block_shapes[path[nops],0]
                            else:
                                err_msg = "Not every operator is unique. Still, \
                                for the unique operator '"+str(nops)+"', there \
                                was no nonunique preceding or subsequent operator \
                                found!"
                                raise ValueError(err_msg)

                    # get BlockSparseMats for this path
                    for nops in range(self.N_ops):
                        if not self.unique_list[nops]:
                            # opMat_array[nops] = op_blocks[pathlist[k][nops]]
                            dummy_BSmatrix = op_mat_list[nops]
                            site_dummy = path[nops]
                            opMat_array[nops] = dummy_BSmatrix.get(site_dummy)
                            if self.withRevTerm:
                                # opMat_rev_array[depth] = op_rev_blocks[pathlist[k][nops]]
                                dummy_BSmatrix = op_mat_rev_list[nops]
                                site_dummy = path[nops]
                                opMat_rev_array[nops] = dummy_BSmatrix.get(site_dummy)


                ### make matrix product by summing over all orbitals
                tmp_sum = 0
                # get vector(=wf) to be multiplied
                for o_i in range(x_norbs[0]):
                    self.mult_vec0[o_i] = bra[bra_start+o_i].conjugate()
                    if self.withRevTerm != 0:
                        self.mult_vec1[o_i] = ket[bra_start + o_i]


                # for each operator, make a vector.matrix product
                for nops in range(self.N_ops):
                    norbs = x_norbs[nops]
                    norbs_next = x_norbs[nops+1]
                    # o_i: index of resulting vector
                    for o_i in range(norbs):
                        # o_j: summation index
                        for o_j in range(norbs_next):
                            tmp_sum += self.mult_vec0[o_j] * \
                                       opMat_array[nops][o_j*norbs+o_i]
                        self.tmp_vec0[o_i] = tmp_sum
                        tmp_sum = 0
                        # the same for reverted term
                        if self.withRevTerm != 0:
                            for o_j in range(norbs_next):
                                if self.check_hermiticity:
                                    tmp_sum += opMat_rev_array[nops][o_j*norbs+o_i].conjugate() \
                                           * self.mult_vec1[o_j]
                                else:
                                    tmp_sum += opMat_rev_array[nops][o_i*norbs_next+o_j] \
                                           * self.mult_vec1[o_j]

                            self.tmp_vec1[o_i] = tmp_sum
                            tmp_sum = 0
                    # store tmp vecs
                    for o_i in range(norbs):
                        self.mult_vec0[o_i] = self.tmp_vec0[o_i]
                        if self.withRevTerm != 0:
                            self.mult_vec1[o_i] = self.tmp_vec1[o_i]
                # final product vector^\dagger.vector, i.e. a scalar product
                for o_j in range(x_norbs[self.N_ops]):
                    tmp_sum +=  self.mult_vec0[o_j] * ket[ket_start+o_j]
                out_data[ipath] = tmp_sum
                tmp_sum = 0
                # scalar product for reverted term
                if self.withRevTerm != 0:
                    for o_j in range(x_norbs[self.N_ops]):
                        tmp_sum += bra[ket_start+o_j].conjugate() * self.mult_vec1[o_j]
                    out_data[ipath] = out_data[ipath] + self.withRevTerm * tmp_sum
                    tmp_sum = 0
                out_data[ipath] = out_data[ipath] * self.const_fac


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
        start = self.auxwhere_list[i]
        end = self.auxwhere_list[i+1]
        if not rev:
            auxhops = self.where_flat[start:end]
        else:
            # revert order of hopping
            auxhops = np.asarray([(hop[1],hop[0]) for hop in self.where_flat[start:end]], dtype=gint_dtype)
        offsets, norbs = kwant.operator._get_all_orbs(auxhops, self._site_ranges)
        return BlockSparseMatrix2(auxhops, offsets, norbs, get_opmat)








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
cdef void _get_orbs2(gint[:, :] site_ranges, gint site,
                    gint *start_orb, gint *norbs):
    """Return the first orbital of this site and the number of orbitals"""
    cdef gint run_idx, first_site, norb, orb_offset, orb
    # Calculate the index of the range that contains the site.
    run_idx = _bisect(site_ranges[:, 0], site) - 1
    first_site = site_ranges[run_idx, 0]
    norb = site_ranges[run_idx, 1]
    orb_offset = site_ranges[run_idx, 2]
    # calculate the slice
    start_orb[0] = orb_offset + (site - first_site) * norb
    norbs[0] = norb

#
#
#
# #99% copy from kwant.operator -> int instead of gint
# def _get_tot_norbs(syst):
#     cdef int _unused, tot_norbs
#     is_infinite_system = isinstance(syst, InfiniteSystem)
#     n_sites = syst.cell_size if is_infinite_system else syst.graph.num_nodes
#     _get_orbs2(np.asarray(syst.site_ranges, dtype=gint_dtype),
#               n_sites, &tot_norbs, &_unused)
#     return tot_norbs


#99% copy from kwant.operator -> int instead of gint
@cython.boundscheck(False)
@cython.wraparound(False)
cdef gint _bisect(gint[:] a, int x):
    "bisect.bisect specialized for searching `site_ranges`"
    cdef gint mid, lo = 0, hi = a.shape[0]
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
