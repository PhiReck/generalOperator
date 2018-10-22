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

from libc.stdlib cimport free, calloc
from cython.view cimport array as cvarray
from inspect import signature

import numpy as np


gint_dtype = np.int32
# cdef extern from "defs.h":
#     ctypedef signed int gint

# supported operations within the `_operate` method
ctypedef enum operation:
    MAT_ELS
    ACT

def _dictwhere_to_auxlists(wherelist, auxwhere_list, dict_list):
    auxpos_list = []
    wherepos = []
    new_pos = 0
    usedSites = []
    for i, xyhops in enumerate(wherelist[:-1]):
        flattened_where_offset = auxwhere_list[i]
        usedSites = []
        for j,hop in enumerate(xyhops):
            if hop[1] in usedSites:
                # check where it occured before
                index = usedSites.index(hop[1])
                # add total offset, i.e. where to find it in auxpos_list
                index += flattened_where_offset
                # copy that position and length
                auxpos_list.append(auxpos_list[index])
                # no change to new_pos, since wherepos was not extended
            else: # site not yet searched for 'neighbors'
                  #in next hopping-level
                usedSites.append(hop[1])
                neighlist = dict_list[i][hop[1]]
                wherepos.extend(neighlist)
                num_neigh = len(neighlist)
                auxpos_list.append((new_pos,num_neigh))
                new_pos += num_neigh

    return np.asarray(wherepos, dtype=gint_dtype), np.asarray(auxpos_list, dtype=gint_dtype)

    # recursive functions for preparing the necessary matrices
    # for a given set of sites and initiate the sum over the orbitals
# cdef void sitesums_recfunc(complex[:] out_data, int* idata, int iSite,
#                            int depth, int ket_start, int bra_start,
#                            int[:,:,:] wherelist, int[:,:,:] wherepos_dicts,
#                            long[:] x_norbs, complex** M_onsite, complex** hopfunc,
#                            BlockSparseMatrix2[:] M_onsite_blocks, BlockSparseMatrix2[:] hopfunc_blocks,
#                            int withRevTerm, complex const_fac, int N_SiteSums,
#                            int* unique_onsite, complex[:] bra, complex[:] ket):
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sitesums_recfunc(complex[:] out_data,
                           int* idata, int iSite,
                            int depth, int ket_start, int bra_start,
                            # int[:,:,:] wherelist,
                            int [:] auxwhere_list,
                            int [:] wherepos_neigh,
                            int [:,:] auxpos_list,
                            # int[:,:,:] wherepos_array,
                            int* x_norbs,
                            complex** M_onsite, complex** hopfunc,
                            BlockSparseMatrix2[:] M_onsite_blocks, BlockSparseMatrix2[:] hopfunc_blocks,
                            int withRevTerm, complex const_fac, int N_SiteSums,
                            int* unique_onsite, complex[:] bra, complex[:] ket
                           ):
    cdef BlockSparseMatrix2 dummyhop_mat
    cdef BlockSparseMatrix2 dummyonsite_mat
    # cdef int nextSite = -1
    cdef int iNextSite, o1_a
    cdef int num_neighbors, startindex_neighbors, auxpos_list_index
    cdef complex sumtmp
    cdef complex orbprod0, orbprod1

    # if N_SiteSums != 1:
    #     nextSite = wherelist[depth-1][iSite][1]

    #recursion end: get last M_onsite and initiate sum over all orbitals
    if depth == (N_SiteSums - 1) or N_SiteSums == 1:
        # fill the last onsite matrix
        if (not unique_onsite[depth]) and N_SiteSums != 1:
            dummyonsite_mat = M_onsite_blocks[depth]
            M_onsite[depth] = dummyonsite_mat.get(iSite)

        # sum over all orbitals for the given set of sites
        sumtmp = 0
        for o1_a in range(x_norbs[0]):
            orbprod0 = bra[bra_start + o1_a].conjugate()
            if withRevTerm != 0:
                orbprod1 = ket[bra_start + o1_a]
            else:
                orbprod1 = 0j
            sumtmp = orbsums_recfunc(sumtmp,
                                     orbprod0, orbprod1, 0,
                                     bra_start, ket_start, o1_a,
                                     withRevTerm,
                                     x_norbs,
                                     N_SiteSums,
                                     M_onsite, hopfunc,
                                     bra,
                                     ket
                                     )
        out_data[idata[0]] = const_fac * sumtmp
        idata[0] = idata[0] + 1

    #recursion: loop over all 'x'-Sites,
    #           get the needed M_x, O_xy and x_norbs,
    #           and call next recursion step
    else:
        assert depth < (N_SiteSums - 1)

        auxpos_list_index = auxwhere_list[depth-1] + iSite
        num_neighbors = auxpos_list[auxpos_list_index][1]
        startindex_neighbors = auxpos_list[auxpos_list_index][0]
        for i in range(num_neighbors):
            iNextSite = wherepos_neigh[startindex_neighbors + i]
        #     sitechains_recfunc(sitechain_data, sitechain_aux, iNextSite, depth+1)
        # for iNextSite in wherepos_array[depth-1][iSite]:
            if not unique_onsite[depth]:
                dummyonsite_mat = M_onsite_blocks[depth]
                M_onsite[depth] = dummyonsite_mat.get(iNextSite)

            dummyhop_mat = hopfunc_blocks[depth]
            hopfunc[depth] = dummyhop_mat.get(iNextSite)
            x_norbs[depth+1] = dummyhop_mat.block_shapes[iNextSite, 1]

            # special case for the forelast loop: get index ket_start
            if depth == (N_SiteSums - 2):
                ket_start = dummyhop_mat.block_offsets[iNextSite, 1]

            sitesums_recfunc(out_data,
                             idata, iNextSite,
                             depth+1,
                             ket_start, bra_start,
                             auxwhere_list,
                             wherepos_neigh,
                             auxpos_list,
                             # wherelist,
                             # wherepos_array,
                             x_norbs,
                             M_onsite, hopfunc, M_onsite_blocks, hopfunc_blocks,
                             withRevTerm, const_fac,
                             N_SiteSums, unique_onsite, bra, ket
                            )


# calculate the sum over all orbitals for a given set of sites
@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex orbsums_recfunc(complex sum, complex orbprod0, complex orbprod1,
                             int orbdepth, int bra_start, int ket_start,
                             int o1_x, int withRevTerm, int* x_norbs,
                             int N_SiteSums, complex** M_onsite, complex** hopfunc,
                             complex[:] bra, complex[:] ket):

    # assert(type(orbdepth) == int)
    # assert(type(o1_x) == int)
    # assert(type(bra_start) == int)
    # assert(type(ket_start) == int)

    cdef int norbs, norbs_next, o1_y, o2_x,
    norbs = x_norbs[orbdepth]
    cdef int neworbdepth = orbdepth + 1
    cdef complex orbprod_tmp0, orbprod_tmp1
    cdef complex sumtmp = sum
    # recursion: multiply M_x*O_xy*orbproduct(sofar), call next step
    if orbdepth < (N_SiteSums - 1):
        # pass
        norbs_next = x_norbs[orbdepth+1]
        for o2_x in range(norbs):
            for o1_y in range(norbs_next):
                orbprod_tmp0 = M_onsite[orbdepth][o1_x*norbs+o2_x] \
                               * hopfunc[orbdepth][o2_x*norbs_next+o1_y]
                orbprod_tmp0 *= orbprod0

                if withRevTerm != 0:
                    orbprod_tmp1 = hopfunc[orbdepth][o2_x*norbs_next
                                   + o1_y].conjugate() \
                                   * M_onsite[orbdepth][o2_x*norbs+o1_x]
                    orbprod_tmp1 *= orbprod1
                # next step
                sumtmp = orbsums_recfunc(sumtmp, orbprod_tmp0, orbprod_tmp1,
                                         neworbdepth, bra_start, ket_start,
                                         o1_y, withRevTerm,
                                         x_norbs,
                                         N_SiteSums, M_onsite, hopfunc, bra,
                                         ket
                                         )

    # recursion end: orbprod(sofar).M_z.ket_z; and sum over the products
    else:
        # assert(orbdepth == (N_SiteSums - 1))
        for o2_x in range(norbs):
            orbprod_tmp0 = M_onsite[orbdepth][o1_x*norbs+o2_x] \
                              * ket[ket_start+o2_x]
            orbprod_tmp0 *= orbprod0

            sumtmp += orbprod_tmp0

            if withRevTerm != 0:
                orbprod_tmp1 = bra[ket_start+o2_x].conjugate() \
                                  * M_onsite[orbdepth][o2_x*norbs+o1_x]
                orbprod_tmp1 *= orbprod1
                sumtmp += withRevTerm * orbprod_tmp1

    return sumtmp








def _where_to_dict_where(syst, wherelist):
    """
    'wherelist' is of the structure [[(a,b),...],[(b,c),...],...,[(x,y),...]],
    with a,b,..,y being Sites of the finalized builder (ie. of type int) or
    unfinalized builder (ie. instance of kwant.builder.Site).
    returns the list of dictionaries, who tell where to find the corresponding
    hoppings of the next site in 'wherelist'.
    """
    # a_list = [i for i in range(len(wherelist[0]))]
    dict_list = []  # [a_list]
    # if type(wherelist[0][0]) == int or isinstance(wherelist[0][0], kwant.builder.Site):
    #     return dict_list
    # else:
    assert(len(wherelist[0][0]) == 2)
    # loop to generate the where_index dicts
    for i in range(len(wherelist)-1):
        dict_list.append({})
        for index, hop in enumerate(wherelist[i+1]):
            if isinstance(hop[0], kwant.builder.Site): # if sites are kwant.builder.Site objects
                site_int = syst.id_by_site[hop[0]]
            # elif type(hop[0]) == int:
            else:
                site_int = hop[0]
            # else:
            #     raise ValueError('Either list of where is of wrong shape or given Sites are neither of type int nor instances of kwant.builder.Site')
            if site_int in dict_list[i].keys():
                dict_list[i][site_int].append(index)
            else:
                dict_list[i][site_int] = [index]
    # # create the last dictionary for the rightmost onsite-terms <- not needed anymore
    # i = len(wherelist)-2
    # dict_list.append({})
    # for index, hop in enumerate(wherelist[i+1]):
    #     if isinstance(hop[1], kwant.builder.Site): # if sites are kwant.builder.Site objects
    #         site_int = syst.id_by_site[hop[0]]
    #     # elif type(hop[1]) == int:
    #     else:
    #         site_int = hop[0]
    #     # else:
    #     #     raise ValueError('Either list of where is of wrong shape or given Sites are neither of type int nor instances of kwant.builder.Site')
    #     if site_int in dict_list[i+1].keys():
    #         dict_list[i+1][site_int].append(index)
    #     else:
    #         dict_list[i+1][site_int] = [index]


    # for better efficiency, create array from dict
    # posarray = wherelist[:-1]
    # # posarray = [None] * (len(wherelist)-1)
    #
    # for i, xyhops in enumerate(wherelist[:-1]):
    #     # posarray[i] = []
    #     for j,hop in enumerate(xyhops):
    #         site_int = syst.id_by_site[hop[1]]
    #         posarray[i][j] = np.asarray(dict_list[i][site_int], dtype=gint_dtype)

    return dict_list






def _create_where_list_from_added_sites(fsyst, intracell_sites, intercell_sites):
    r"""Creates where list from the sitelists 'intracell_sites' and 'intercell_sites' for (lead) energy current calculation.

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






def _create_sitechains_output(where, auxwhere_list, auxpos_list, wherepos_neigh, N_depth):
    """
    Gives back a list, which has the same sructure as the genOp output_data.
    This list tells you, which site-combination was used to create the corres-
    poding output_data
    """
    #recursive function to get all combinations of sites of the sums
    def sitechains_recfunc(sitechain_data, sitechain_aux, int iSite, int depth):
        # add the next site to the chain
        nextSite = where[depth-1][iSite][1]
        sitechain_aux[depth] = nextSite
        #recursion end: copy sitechain_aux to sitechain_data
        if depth == (N_depth - 1):
            sitechain_data.append(sitechain_aux[:]) #sliced for copy, avoid reference
        #recursion: call the next step
        else:
            assert depth < (N_depth - 1)
            auxpos_list_index = auxwhere_list[depth-1] + iSite
            for i in range(auxpos_list[auxpos_list_index][1]):
                iNextSite = wherepos_neigh[i + auxpos_list[auxpos_list_index][0]]
                sitechains_recfunc(sitechain_data, sitechain_aux, iNextSite, depth+1)
        # end recursive function


    sitechain_data = []
    sitechain_aux = [None] * N_depth
    # sum over all 'a'-Sites and initialize first step (for 'b'-Sites)
    for ia in range(len(where[0])):
        sitechain_aux[0] = where[0][ia][0]
        if N_depth > 1:
            sitechains_recfunc(sitechain_data, sitechain_aux, ia, 1)

    return np.asarray(sitechain_data)


cdef class generalOperator:
    r"""
    KOMMENTARE MUESSEN NOCH ANGEPASST WERDEN!!
    An operator for calculating general currents with arbitrary hopping.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Current.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    arbit_hop_func: tbd
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
    Mostly needed for explicite time-dependendence of hoppings for heat current in a lead:
    :math:`\psi_j^\dagger  \partial H_{ij} /\partial t \psi_i
    """

    ### TODO: For which of those objects does it make sense to be "global"?
    cdef public int check_hermiticity, sum, N_SiteSums, withRevTerm, out_data_length
    cdef public complex const_fac
    cdef public object syst, onsite, _onsite_params_info
    cdef public object hopOperator # wherepos_dicts,
    cdef public object _bound_onsite_list, _bound_hopfunc_list, wherelist
    # cdef public int[:, :, :]  wherelist
    cdef public int[:, :, :] wherepos_array
    cdef public int[:, :] _site_ranges, auxpos_list, sitechains #, where
    cdef public int[:] auxwhere_list, wherepos_neigh
    cdef public BlockSparseMatrix2[:] hopfunc_blocks, M_onsite_blocks

    # cdef public int numbercalls

    @cython.embedsignature
    def __init__(self, syst, onsite_list, hopOp_list, in_wherelist, where_pos_idx= None,
                 withRevTerm=0, const_fac=1, *, check_hermiticity=False, sum=False):
        # self.numbercalls = 0
        assert len(onsite_list) == len(hopOp_list)+1
        if len(onsite_list) > 1:
            assert len(onsite_list) == len(in_wherelist)+1
        else:
            assert len(in_wherelist) == 1  # where = [ [site1,site2,...] ]
        self.N_SiteSums = len(onsite_list)
        self.syst = syst
        self._site_ranges = np.asarray(syst.site_ranges, dtype=gint_dtype)
        self.withRevTerm = withRevTerm
        self.const_fac = const_fac
        self.check_hermiticity = check_hermiticity
        self.sum = sum

        # get hopping-lists in needed format
        # self.wherelist = np.empty_like(in_wherelist)
        _wherelist = in_wherelist[:]
        if self.N_SiteSums == 1:  # special case of "Density" (ie. bra.M.ket)
            try:
                if type(_wherelist[0][0]) == int:  # _normalize... does not accept ints
                    _wherelist[0] = [syst.sites[i] for i in _wherelist[0]]
            except:
                pass
            _wherelist[0] = kwant.operator._normalize_site_where(syst, in_wherelist[0])
            where_pos_idx = [[i for i in range(len(_wherelist[0]))]]

        # all other cases except "Density": hoppings are given instead of sites
        for i in range(self.N_SiteSums - 1):
            try:  # maybe _wherelist[0]==None, in which case an Error occurs
                if type(_wherelist[0][0]) == int:  # _normalize... does not accept ints
                    _wherelist[i] = [(syst.sites[hop[0]], syst.sites[hop[1]]) for hop in _wherelist[i]]
            except:
                pass
            _wherelist[i] = kwant.operator._normalize_hopping_where(syst, in_wherelist[i])

        self.wherelist = np.asarray(_wherelist)

        # create dictionary to find needed hoppings or use predefined one
        if self.N_SiteSums > 2:
            if where_pos_idx:
                dict_list = where_pos_idx
            # get dictionary for "connected hoppings" if not yet defined
            else:
                dict_list = _where_to_dict_where(syst, in_wherelist)
            # for calculation reasons, store the data of the dictionary in a
            # 1d array and use two auxiliary lists for bookkeeping
            _auxwhere_list = np.zeros(len(in_wherelist), dtype=gint_dtype)
            next_wherelist_start = 0
            for i, xyhops in enumerate(in_wherelist):
                _auxwhere_list[i] = next_wherelist_start
                next_wherelist_start += len(xyhops)
            self.auxwhere_list = _auxwhere_list
       #     # create lists to help finding the positions in 'where_list' of the
       #     # neighbors (of a given Site) in the next hopping level
            self.wherepos_neigh, self.auxpos_list = _dictwhere_to_auxlists(self.wherelist, self.auxwhere_list, dict_list)
        else:
            self.auxwhere_list = np.asarray([0,len(self.wherelist[0])], dtype=gint_dtype)
            #fake auxlists
            self.wherepos_neigh = np.asarray([0], dtype=gint_dtype)
            self.auxpos_list = np.asarray([[0,0]], dtype=gint_dtype)


        # print('dict_list=', dict_list)
        # print('auxwhere_list[0]=', self.auxwhere_list[0])
        # print('auxwhere_list[1]=', self.auxwhere_list[1])
        # print('wherepos_neigh=', self.wherepos_neigh)
        # print('auxpos_list=', self.auxpos_list)

        # create a list with all possible combinations of the given sites;
        # list ordering is the same as of the output_data
        if self.N_SiteSums > 2:
            self.sitechains = _create_sitechains_output(_wherelist, self.auxwhere_list, self.auxpos_list, self.wherepos_neigh, self.N_SiteSums)
            # print('self.sitechains = ', self.sitechains)
        # if DENSITY or Current:
        # data_length = number of Sites or Hoppings in where
        elif self.N_SiteSums == 2:
            self.sitechains = np.asarray([[hop[0],hop[1]] for hop in self.wherelist[0]])
            # self.out_data_length = len(_wherelist[0])
        else:
            self.sitechains = self.wherelist[0] #.tolist()
            # self.out_data_length = len(_wherelist[0])

        self.out_data_length = len(self.sitechains)

        # hopping operators
        # define unit hopping in case needed
        def unit_hop(a, b, *args, **params):
            cdef int a_norbs, b_norbs, dummy
            _get_orbs(self._site_ranges, a, &dummy, &a_norbs)
            _get_orbs(self._site_ranges, a, &dummy, &b_norbs)
            retmat = np.eye(a_norbs, b_norbs,dtype=complex)
            if a==b:
                for ia in range(a_norbs):
                    retmat[ia,ia] = 1+0j
            return retmat
        # replace 'default'-values 'h' and 'unit' by functions
        self.hopOperator = [None] * (self.N_SiteSums-1)
        for i, hopfunc in enumerate(hopOp_list):
            try:
            #check if hoppingOperator == Hamiltonian
                if hopfunc in ('H','h'):
                    hopfunc = self.syst.hamiltonian
                #check if hoppingOperator == unit matrix
                elif hopfunc in (1,'unit'):
                    hopfunc = unit_hop
            except TypeError:
                pass
            self.hopOperator[i] = hopfunc

        # onsite operators
        # define onsite Hamiltonian in case onsite operator is Hamiltonian
        ### TODO: Check if it is already implemented in kwant
        def onsiteHamil(a, *args, **params):
            if type(a) == kwant.builder.Site:
                a = syst.id_by_site[a]
            assert(type(a) == int)
            return syst.hamiltonian(a, a, *args, params=params)
        # replace 'default'-values ('h' and 'H') by functions
        self.onsite = [None] * self.N_SiteSums
        self._onsite_params_info = [None] * self.N_SiteSums
        for i, onsitefunc in enumerate(onsite_list):
            #check if onsite == Hamiltonian
            try:
                if onsitefunc == 'h' or onsitefunc == 'H':
                    onsitefunc = onsiteHamil
            except TypeError:
                pass
            self.onsite[i], self._onsite_params_info[i] = \
                            kwant.operator._normalize_onsite(syst, onsitefunc, check_hermiticity)



        # where to store the BlockSparseMatrices
            # COMMENT: The following is a workaround to create an array of
            # BlocksparseMatrices instead of a python list for faster access.
            # To create this array, the Blocksparsemats have to be initialized,
            # here by dummy_Blocksparse
        def zero(*args,**kwargs):
            matrix = ta.matrix
            mat = matrix(0j, complex)
            return mat

        cdef BlockSparseMatrix2 dummy_Blocksparse

        fakelist = np.asarray([(1,1)], dtype=gint_dtype)
        dummy_Blocksparse = BlockSparseMatrix2(fakelist,fakelist,fakelist,zero)
        dummy_list = [dummy_Blocksparse]*(self.N_SiteSums-1)
        if self.N_SiteSums > 1:
            self.hopfunc_blocks = np.asarray(dummy_list)
            # self._bound_hopfunc_list = np.asarray(dummy_list)
        dummy_list.append(dummy_Blocksparse)
        self.M_onsite_blocks = np.asarray(dummy_list)
        # self._bound_onsite_list = np.asarray(dummy_list)

        self._bound_onsite_list = [None]*self.N_SiteSums
        self._bound_hopfunc_list = [None]*(self.N_SiteSums-1)



    # If one wants to know which output-data belongs to which site combination
    @cython.embedsignature
    def get_sitechains(self):
        return self.sitechains

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        # self.numbercalls = self.numbercalls + 1
        r"""Return the matrix elements of the operator.

        An operator ``A`` can be called like

            >>> A(psi)

        to compute the expectation value :math:`\bra{ψ} A \ket{ψ}`,
        or like

            >>> A(phi, psi)

        to compute the matrix element :math:`\bra{φ} A \ket{ψ}`.

        If ``sum=True`` was provided when constructing the operator, then
        a scalar is returned. If ``sum=False``, then a vector is returned.
        The vector is defined over the sites of the system if the operator
        is a `~kwant.operator.Density`, or over the hoppings if it is a
        `~kwant.operator.Current` or `~kwant.operator.Source`. By default,
        the returned vector is ordered in the same way as the sites
        (for `~kwant.operator.Density`) or hoppings in the graph of the
        system (for `~kwant.operator.Current` or `~kwant.operator.Density`).
        If the keyword parameter ``where`` was provided when constructing
        the operator, then the returned vector is instead defined only over
        the sites or hoppings specified, and is ordered in the same way
        as ``where``.

        Alternatively stated, for an operator :math:`Q_{iαβ}`, ``bra``
        :math:`φ_α` and ``ket`` :math:`ψ_β` this computes
        :math:`q_i = ∑_{αβ} φ^*_α Q_{iαβ} ψ_β` if ``self.sum`` is False,
        otherwise computes :math:`q = ∑_{iαβ} φ^*_α Q_{iαβ} ψ_β`. where
        :math:`i` runs over all sites or hoppings, and
        :math:`α` and :math:`β` run over all the degrees of freedom.

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
        # Sanity check to be thought of for binding only certain Operators
        # if (self._bound_onsite or self._bound_hamiltonian) and (args or params):
            # raise ValueError("Extra arguments are already bound to this "
            #                  "operator. You should call this operator "
            #                  "providing neither 'args' nor 'params'.")
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
    def bind(self, onsites_tobebound=[], hoppings_tobebound=[], *,
             args=(), params=None):
        """Bind the given arguments to this operator.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called or when using the ``act`` method.
        """
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")
        # generic creation of new instance
        cls = self.__class__
        q = cls.__new__(cls)

        q.N_SiteSums = self.N_SiteSums
        q.syst = self.syst
        q._site_ranges = self._site_ranges
        q.withRevTerm = self.withRevTerm
        q.const_fac = self.const_fac
        q.check_hermiticity = self.check_hermiticity
        q.sum = self.sum
        q.wherelist = self.wherelist
        q.sitechains = self.sitechains
        q.out_data_length = self.out_data_length
        q.auxwhere_list = self.auxwhere_list
        q.wherepos_neigh = self.wherepos_neigh
        q.auxpos_list = self.auxpos_list
        q.hopOperator = self.hopOperator
        q.onsite = self.onsite
        q._onsite_params_info = self._onsite_params_info
        q._bound_onsite_list = self._bound_onsite_list
        q._bound_hopfunc_list = self._bound_hopfunc_list
        q.hopfunc_blocks = self.hopfunc_blocks
        q.M_onsite_blocks = self.M_onsite_blocks

        # if onsites_tobebound[0] in None and hoppings_tobebound[0] is None:
        #     onsites_tobebound = 'all'
        #     hoppings_tobebound = 'all'

        if onsites_tobebound == 'all':
            onsites_tobebound = list(range(self.N_SiteSums))
        if hoppings_tobebound == 'all':
            hoppings_tobebound = list(range(self.N_SiteSums - 1))

        for index in onsites_tobebound:
            if callable(self.onsite[index]):
                q._bound_onsite_list[index] = self._eval_onsites(index, args, params)

        for index in hoppings_tobebound:
            q._bound_hopfunc_list[index] = self._eval_hop_func(index, args, params)

        return q

    #100% copy of kwant.operators.Current
    # @cython.embedsignature
    # def bind(self, args=(), *, params=None):
    #     """Bind the given arguments to this operator.
    #
    #     Returns a copy of this operator that does not need to be passed extra
    #     arguments when subsequently called or when using the ``act`` method.
    #     """
    #     q = super().bind(args, params=params)
    #     q._bound_hamiltonian = self._eval_hamiltonian(args, params)
    #     return q

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        """Do an operation with the operator.
            TO BE CHECKED!!!
        Parameters
        ----------
        out_data : ndarray
         Output array, zero on entry. On exit should contain the required
         data.  What this means depends on the value of `op`, as does the
         length of the array.
        bra, ket : ndarray
         Wavefunctions defined over all the orbitals of the system.
         If `op` is `ACT` then `bra` is None.
        args : tuple
         The extra arguments to the Hamiltonian value functions and
         the operator ``onsite`` function. Mutually exclusive with 'params'.
        op : operation
         The operation to perform.
         `MAT_ELS`: calculate matrix elements between `bra` and `ket`
         `ACT`: act on `ket` with the operator
        params : dict, optional
         Dictionary of parameter names and their values. Mutually exclusive
         with 'args'.
        """

        cdef int i
        # prepare matrices for onsite and arbitrary hopping functions
        cdef complex[:, :] _tmp_mat
        #could be done in __init__, but problems with coding!
        cdef int * unique_onsite = <int*>calloc(self.N_SiteSums, sizeof(int))
        for i in range(self.N_SiteSums):
            unique_onsite[i] = not callable(self.onsite[i])
        cdef complex **M_onsite = \
                          <complex **>calloc(self.N_SiteSums, sizeof(complex*))
        cdef complex **hopfunc = \
                      <complex **>calloc((self.N_SiteSums-1), sizeof(complex*))



        cdef BlockSparseMatrix2 dummy_mat_hopval
        cdef BlockSparseMatrix2 dummy_mat_onsiteval
        # create necessary onsite BlockSparseMatrices if not unique
        #could be done in __init__? NO! mats could change over time
        for i in range(self.N_SiteSums):
            if unique_onsite[i]:
                _tmp_mat = self.onsite[i]
                M_onsite[i] = <complex*> &_tmp_mat[0, 0]
            elif self._bound_onsite_list[i]:
                dummy_mat_onsiteval = self._bound_onsite_list[i]
                self.M_onsite_blocks[i] = dummy_mat_onsiteval
            else:
                dummy_mat_onsiteval = self._eval_onsites(i, args, params)
                self.M_onsite_blocks[i] = dummy_mat_onsiteval

        # create necessary hopping BlockSparseMatrices
        for i in range(self.N_SiteSums-1):
            if self._bound_hopfunc_list[i]:
                dummy_mat_hopval = self._bound_hopfunc_list[i]
                self.hopfunc_blocks[i] = dummy_mat_hopval
            else:
                dummy_mat_hopval = self._eval_hop_func(i, args, params)
                self.hopfunc_blocks[i] = dummy_mat_hopval


        # main loop
        #could be done in __init__
        cdef int * x_norbs = <int*>calloc(self.N_SiteSums, sizeof(int))
        # cdef int[:] x_norbs
        # x_norbs = np.zeros(self.N_SiteSums, dtype=np.dtype("i"))
        cdef int ket_start = 0
        cdef int bra_start, a_s  # indices for wave function
        cdef int ia, iNextSite  # where given Site is to be found in wherelist
        cdef int a_norbs, norbs, norbs_next  #number of orbitals
        cdef int a, b, nextSite  # site IDs
        cdef int o1_a, o1_x, o2_x, o1_y  # orbit IDs
        cdef complex orbprod_tmp[2]  # for intermediate product of orbitals
        cdef complex orbprod[2]  # for products of orbitals
        cdef complex sumtmp  # for the sum of orbitals products
        cdef int idata = 0

        # # recursive functions for preparing the necessary matrices
        # # for a given set of sites and initiate the sum over the orbitals
        # def sitesums_recfunc(out_data, int iSite, int depth, int ket_start = 0):
        #
        #     cdef BlockSparseMatrix2 dummyhop_mat
        #     cdef BlockSparseMatrix2 dummyonsite_mat
        #
        #     # get site ID of site connected to previous site
        #     if self.N_SiteSums != 1:
        #         nextSite = self.wherelist[depth-1][iSite][1]
        #
        #     # #recursion end: get last M_onsite and initiate sum over all orbitals
        #     if depth == (self.N_SiteSums - 1) or self.N_SiteSums == 1:
        #         # if needed: fill the last onsite matrix
        #         if (not unique_onsite[depth]) and self.N_SiteSums != 1:
        #             dummyonsite_mat =  M_onsite_blocks[depth]
        #             M_onsite[depth] = dummyonsite_mat.get(iSite)
        #
        #         # sum over all orbitals for the given set of sites
        #         sumtmp = 0
        #         for o1_a in range(x_norbs[0]):
        #             orbprod[0] = bra[bra_start + o1_a].conjugate()
        #             if self.withRevTerm != 0:
        #                 orbprod[1] = ket[bra_start + o1_a]
        #         sumtmp = orbsums_recfunc(sumtmp, orbprod, o1_a, 0, bra_start, ket_start)
        #         # append result for one Sites-combination to output_data
        #         out_data.append(self.const_fac * sumtmp)
        #
        #     #recursion:
        #     else:
        #         assert depth < (self.N_SiteSums - 1)
        #         # loop over all 'x'-Sites,
        #         # i.e. Sites connected to the previous site (iSite)
        #         for iNextSite in self.wherepos_dicts[depth][nextSite]:
        #             # get M_x
        #             if not unique_onsite[depth]:
        #                 dummyonsite_mat =  M_onsite_blocks[depth]
        #                 M_onsite[depth] = dummyonsite_mat.get(iNextSite)
        #             # get O_xy
        #             dummyhop_mat = hopfunc_blocks[depth]
        #             hopfunc[depth] = dummyhop_mat.get(iNextSite)
        #             # get x_norbs
        #             x_norbs[depth+1] = dummyhop_mat.block_shapes[iNextSite, 1]
        #
        #             # special case for the forelast loop: get index ket_start
        #             if depth == (self.N_SiteSums - 2):
        #                 ket_start = dummyhop_mat.block_offsets[iNextSite, 1]
        #
        #             # call next recursion step
        #             sitesums_recfunc(out_data, iNextSite, depth+1, ket_start=ket_start)


        # # calculate the sum over all orbitals for a given set of sites
        # def orbsums_recfunc(sumtmp, orbprod, o1_x, orbdepth, bra_start=0, ket_start=0):
        #     assert(type(orbdepth) == int)
        #     assert(type(o1_x) == int)
        #     assert(type(bra_start) == int)
        #     assert(type(ket_start) == int)
        #
        #     norbs = x_norbs[orbdepth]
        #     # recursion: multiply M_x*O_xy*orbproduct(sofar), call next step
        #     if orbdepth < (self.N_SiteSums - 1):
        #         norbs_next = x_norbs[orbdepth+1]
        #         for o2_x in range(norbs):
        #             for o1_y in range(norbs_next):
        #                 # M_x*O_xy
        #                 orbprod_tmp[0] = M_onsite[orbdepth][o1_x*norbs+o2_x] \
        #                                * hopfunc[orbdepth][o2_x*norbs_next+o1_y]
        #                 # times orbproduct(sofar)
        #                 orbprod_tmp[0] *= orbprod[0]
        #                 # if complex conjugate term is needed
        #                 if self.withRevTerm != 0:
        #                     orbprod_tmp[1] = hopfunc[orbdepth][o2_x*norbs_next
        #                                    + o1_y].conjugate() \
        #                                    * M_onsite[orbdepth][o2_x*norbs+o1_x]
        #                     orbprod_tmp[1] *= orbprod[1]
        #                 # call next step
        #                 sumtmp = orbsums_recfunc(sumtmp, orbprod_tmp,
        #                                          o1_y, orbdepth+1,
        #                                          bra_start, ket_start)
        #
        #     # recursion end: orbprod(sofar).M_z.ket_z; and sum over the products
        #     else:
        #         assert(orbdepth == (self.N_SiteSums - 1))
        #         for o2_x in range(norbs):
        #             # dot product: M_z . ket_z;
        #             orbprod_tmp[0] = M_onsite[orbdepth][o1_x*norbs+o2_x] \
        #                               * ket[ket_start+o2_x]
        #             # times results so far
        #             orbprod_tmp[0] *= orbprod[0]
        #             # summing up result
        #             sumtmp += orbprod_tmp[0]
        #             # if complex conjugate term is needed
        #             if self.withRevTerm != 0:
        #                 orbprod_tmp[1] = bra[ket_start+o2_x].conjugate() \
        #                                   * M_onsite[orbdepth][o2_x*norbs+o1_x]
        #                 orbprod_tmp[1] *= orbprod[1]
        #                 sumtmp += self.withRevTerm * orbprod_tmp[1]
        #
        #     return sumtmp
        # cdef int[:,:,:] fakeDict
        # fakeDict = np.zeros(27,dtype=np.dtype("i")).reshape((3,3,3))
        # out_data_aux = []
        if op == MAT_ELS:
            for ia in range(self.auxwhere_list[1]):
                # if only bra_a M_a ket_a:
                # if only Density: bra_a M_a ket_a:
                # if self.N_SiteSums == 1:
                    # try:  # in where either (fake-)hoppings or directly sites
                    #     a = self.wherelist[0][ia][0]
                    # except TypeError:
                    # print('type(self.wherelist)=',type(self.wherelist))
                    # print('type(self.wherelist[0])=',type(self.wherelist[0]))
                    # get the first onsite matrix if necessary (ie not unique)
                    # if not unique_onsite[0]:
                    #     dummy_mat_onsiteval = M_onsite_blocks[0]
                    #     M_onsite[0] = dummy_mat_onsiteval.get(ia)
                    # a = self.wherelist[0,ia,0]
                    # # a = self.where[ia, 0]
                    # _get_orbs(self._site_ranges, a, &a_s, &a_norbs)
                    # x_norbs[0] = a_norbs
                    # bra_start = a_s
                    # ket_start = bra_start
                    #  # sum over all orbitals for the given set of sites
                    # norbs = a_norbs
                    # orbprod[1] = 0
                    # sumtmp = 0
                    # for o1_a in range(x_norbs[0]):
                    #     o1_x = o1_a
                    #     orbprod[0] = bra[bra_start + o1_a].conjugate()
                    #     if self.withRevTerm != 0:
                    #         orbprod[1] = ket[bra_start + o1_a]
                    #
                    #     for o2_x in range(norbs):
                    #         orbprod_tmp[0] = \
                    #                 M_onsite[0][o1_x*norbs+o2_x] \
                    #                 * ket[ket_start+o2_x]
                    #         orbprod_tmp[0] *= orbprod[0]
                    #
                    #         sumtmp += orbprod_tmp[0]
                    #
                    #         if self.withRevTerm != 0:
                    #             orbprod_tmp[1] = bra[ket_start+o2_x].conjugate() \
                    #                               * M_onsite[0][o2_x*norbs+o1_x]
                    #             orbprod_tmp[1] *= orbprod[1]
                    #             sumtmp += self.withRevTerm * orbprod_tmp[1]
                    #
                    # out_data[idata] = self.const_fac * sumtmp
                    # idata = idata + 1


                # get the first onsite matrix if necessary (ie not unique)
                if not unique_onsite[0]:
                    dummy_mat_onsiteval = self.M_onsite_blocks[0]
                    M_onsite[0] = dummy_mat_onsiteval.get(ia)

                if self.N_SiteSums == 1:
                    # try:  # in where either (fake-)hoppings or directly sites
                    #     a = self.wherelist[0][ia][0]
                    # except TypeError:
                    #     a = self.wherelist[0][ia]
                    a = self.wherelist[0,ia,0]
                    # a = self.where[ia, 0]
                    _get_orbs(self._site_ranges, a, &a_s, &a_norbs)
                    x_norbs[0] = a_norbs
                    bra_start = a_s
                    ket_start = bra_start
                    # second M_onsite_blocks is Just Fake, hopfunc_blocks is not defined here
                    sitesums_recfunc(out_data,
                                     &idata, ia
                                     , 1, ket_start, bra_start,
                                     # self.wherelist,
                                     # fakeDict,
                                     self.auxwhere_list, 
                                     self.wherepos_neigh,
                                     self.auxpos_list,
                                     # #self.wherepos_dicts[1:],
                                     x_norbs,
                                     M_onsite,
                                     hopfunc,
                                     self.M_onsite_blocks,
                                     self.M_onsite_blocks,
                                     self.withRevTerm, self.const_fac,
                                     self.N_SiteSums, unique_onsite, bra, ket
                                     )

                else:  # self.N_SiteSums > 1:
                    # get the first hopfunc matrix
                    dummy_mat_hopval = self.hopfunc_blocks[0]
                    hopfunc[0] =  dummy_mat_hopval.get(ia)
                    # get wf start index and number of orbitals
                    bra_start = dummy_mat_hopval.block_offsets[ia, 0]
                    x_norbs[0] = dummy_mat_hopval.block_shapes[ia, 0]
                    x_norbs[1] = dummy_mat_hopval.block_shapes[ia, 1]
                    # if N_SiteSums == 2, we already know the ket_start index:
                    if self.N_SiteSums == 2:
                        ket_start = dummy_mat_hopval.block_offsets[ia, 1]
                    # call recursive function to get all needed Mats for given Sites
                    sitesums_recfunc(out_data,
                                     &idata, ia, 1, ket_start, bra_start,
                                     # self.wherelist,
                                     # fakeDict,
                                     self.auxwhere_list,
                                     self.wherepos_neigh,
                                     self.auxpos_list,
                                     #self.wherepos_dicts[1:],
                                     x_norbs,
                                     M_onsite,
                                     hopfunc,
                                     self.M_onsite_blocks,
                                     self.hopfunc_blocks,
                                     self.withRevTerm, self.const_fac,
                                     self.N_SiteSums, unique_onsite, bra, ket
                                     )
                # END OF LOOP OVER ALL SITE COMBINATIONS

            # if self.N_SiteSums > 1:
            # # gather data for output
            assert idata == self.out_data_length
            # assert len(out_data_aux) == self.out_data_length
            # for i in range(self.out_data_length):
            #     out_data[i] = out_data_aux[i]


        elif op == ACT:
            raise NotImplementedError()

        free(M_onsite)
        free(hopfunc)
        free(unique_onsite)
        free(x_norbs)


    cdef BlockSparseMatrix2 _eval_hop_func(self, int i, args, params):
        """Evaluate the onsite matrices on all elements of `where`"""

        assert not (args and params)
        params = params or {}
        matrix = ta.matrix

        hop_func = self.hopOperator[i]

        check_hermiticity = self.check_hermiticity
         #XXX: Checks for sanity of 'hop_func' are missing
         # required, defaults, takes_kw = self._onsite_params_info
         # invalid_params = set(params).intersection(set(defaults))
         # if invalid_params:
         #     raise ValueError("Parameters {} have default values "
         #                      "and may not be set with 'params'"
         #                      .format(', '.join(invalid_params)))
         #
         # if params and not takes_kw:
         #     params = {pn: params[pn] for pn in required}

        def get_hop_func(a, a_norbs, b, b_norbs):
            mat = matrix(hop_func(a, b, *args, params=params), complex)
            _check_ham2(mat, hop_func, args, params,
                       a, a_norbs, b, b_norbs, check_hermiticity)
            return mat

        offsets, norbs = kwant.operator._get_all_orbs(self.wherelist[i], self._site_ranges)
        return BlockSparseMatrix2(self.wherelist[i], offsets, norbs, get_hop_func)


     #100% copy of kwant.operators.Current
    cdef BlockSparseMatrix2 _eval_onsites(self, int i, args, params):
        """Evaluate the onsite matrices on all elements of `where`"""
        assert callable(self.onsite[i])
        assert not (args and params)
        params = params or {}
        matrix = ta.matrix
        onsite = self.onsite[i]
        check_hermiticity = self.check_hermiticity

        required, defaults, takes_kw = self._onsite_params_info[i]
        invalid_params = set(params).intersection(set(defaults))
        if invalid_params:
            raise ValueError("Parameters {} have default values "
                             "and may not be set with 'params'"
                             .format(', '.join(invalid_params)))

        if params and not takes_kw:
            params = {pn: params[pn] for pn in required}

        def get_onsite(a, a_norbs, b, b_norbs):
            mat = matrix(onsite(a, *args, **params), complex)
            _check_onsite2(mat, a_norbs, check_hermiticity)
            return mat
        if  i < (self.N_SiteSums - 1) or self.N_SiteSums == 1:
            offsets, norbs = kwant.operator._get_all_orbs(self.wherelist[i], self._site_ranges)
            return  BlockSparseMatrix2(self.wherelist[i], offsets, norbs, get_onsite)
        else:
            assert i == (self.N_SiteSums - 1)
            # get_onsite only uses the first element of hopping -> for the last
            # onsite-operator, we need the second site in hopping to become now
            # the first one
            auxlist = np.asarray([(hop[1],hop[0]) for hop in self.wherelist[i-1]])
            offsets, norbs = kwant.operator._get_all_orbs(auxlist, self._site_ranges)
            return  BlockSparseMatrix2(auxlist, offsets, norbs, get_onsite)


    # def __getstate__(self):
    #     return (
    #         (self.check_hermiticity, self.sum),
    #         (self.syst, self.onsite, self._onsite_params_info),
    #         tuple(map(np.asarray, (self.where, self._site_ranges))),
    #         (self._bound_onsite_list, self._bound_hp[]),
    #     )
    #
    # def __setstate__(self, state):
    #     ((self.check_hermiticity, self.sum),
    #      (self.syst, self.onsite, self._onsite_params_info),
    #      (self.where, self._site_ranges),
    #      (self._bound_onsite, self._bound_hamiltonian),
    #     )
    #





# cdef class Density(_LocalOperator):
cdef class Density(generalOperator):
    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *, check_hermiticity=True, sum=False):

        onsitelist=[onsite]
        hopOplist=[]

        super().__init__(syst, onsitelist, hopOplist, [where], withRevTerm=0,
                      const_fac=1, check_hermiticity=check_hermiticity, sum=sum)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # def __call__(self, bra, ket=None, args=(), *, params=None):
    #     ret = self.density(bra, ket=ket, args=args, params=params)
    #     return ret
    #
    # def get_sitechains(self):
    #     return self.density.get_sitechains()
    #
    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return super().bind(onsites_tobebound=[0], args=args, params=params)


# cdef class Current(kwant.operator._LocalOperator):
cdef class Current(generalOperator):
    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *,
                 check_hermiticity=True, sum=False):

        onsitelist=[onsite,1]
        hopOplist=['h']

        super().__init__(syst, onsitelist, hopOplist, [where], withRevTerm=-1, const_fac=-1j, check_hermiticity=check_hermiticity, sum=sum)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # def __call__(self, bra, ket=None, args=(), *, params=None):
    #     return self.current(bra, ket=ket, args=args, params=params)
    #
    # def get_sitechains(self):
    #     return self.source.get_sitechains()
    #
    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return super().bind(onsites_tobebound='all',
                                 hoppings_tobebound='all',
                                 args=args, params=params)


# cdef class Source(_LocalOperator):
cdef class Source(generalOperator):
    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *, check_hermiticity=True, sum=False):
        where_norm = where[:]
        try:
            if type(where[0]) == int:  # _normalize... does not accept ints
                where_norm = [syst.sites[i] for i in where_norm[0]]
        except:
            pass
        where_norm = kwant.operator._normalize_site_where(syst, where)
        where_norm = [[(syst.sites[site], syst.sites[site]) for site in where_norm]]

        onsitelist=[onsite,'h']
        hopOplist=[1]

        super().__init__(syst, onsitelist, hopOplist, where_norm, withRevTerm=-1, const_fac=-1j, check_hermiticity=check_hermiticity, sum=sum)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # def __call__(self, bra, ket=None, args=(), *, params=None):
    #     ret = self.source(bra, ket=ket, args=args, params=params)
    #     return ret
    #
    # def get_sitechains(self):
    #     return self.source.get_sitechains()

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        return super().bind(onsites_tobebound='all',
                                hoppings_tobebound='all',
                                args=args, params=params)


# cdef class CurrentHdot(_LocalOperator):
cdef class CurrentWithArbitHop(generalOperator):
    @cython.embedsignature
    def __init__(self, syst, onsite=1, arbit_hop_func=0, where=None, *,
                 check_hermiticity=True, sum=False, small_h=0.01):

        onsitelist=[onsite,1]
        hopOplist=[arbit_hop_func]

        super().__init__(syst, onsitelist, hopOplist, [where], withRevTerm=+1, const_fac=1, check_hermiticity=check_hermiticity, sum=sum)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # def __call__(self, bra, ket=None, args=(), *, params=None):
    #     ret = self.currentWithArbitHop(bra, ket=ket, args=args, params=params)
    #     return ret
    #
    # def get_sitechains(self):
    #     return self.source.get_sitechains()
    #
    # @cython.embedsignature
    # def bind(self, onsites_tobebound=[None], hoppings_tobebound=[None],
    #          args=(),params=None):
    #     return super().bind(onsites_tobebound,
    #                                          hoppings_tobebound,
    #                                          args=args, params=params)

# cdef class offEnergyCurrent(_LocalOperator):
cdef class offEnergyCurrent(generalOperator):
    @cython.embedsignature
    def __init__(self, syst, where, *, check_hermiticity=True, sum=True):

        onsitelist=[1,1,1]
        hopOplist=['h','h']

        super().__init__(syst, onsitelist, hopOplist, where, withRevTerm=-1, const_fac=1j, check_hermiticity=check_hermiticity, sum=sum)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # def __call__(self, bra, ket=None, args=(), *, params=None):
    #     ret = self.offEnergyCurrent(bra, ket=ket, args=args, params=params)
    #     return ret
    #
    # def get_sitechains(self):
    #     return self.source.get_sitechains()

    # @cython.embedsignature
    # def bind(self, onsites_tobebound=[None], hoppings_tobebound=[None],
    #          args=(),params=None):
    #     return self.offEnergyCurrent.bind(onsites_tobebound, hoppings_tobebound,
    #                                       args=args, params=params)









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






#_herm_msg = ('{0} matrix is not hermitian, use the option '
#             '`check_hermiticity=True` if this is intentional.')
#_shape_msg = ('{0} matrix dimensions do not match '
#      'the declared number of orbitals')

cdef int _check_ham2(complex[:, :] H, ham, args, params,
                    int a, int a_norbs, int b, int b_norbs,
                    int check_hermiticity) except -1:
    "Check Hamiltonian matrix for correct shape and hermiticity."
    if H.shape[0] != a_norbs and H.shape[1] != b_norbs:
        raise UserCodeError(kwant.operator._shape_msg.format('Hamiltonian'))
    if check_hermiticity:
        # call the "partner" element if we are not on the diagonal
        H_conj = H if a == b else ta.matrix(ham(b, a, *args, params=params),
                                                complex)
        if not _is_herm_conj2(H_conj, H):
            raise ValueError(kwant.operator._herm_msg.format('Hamiltonian'))
    return 0


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


# if possible, use _kwant.operator._get_orbs instead


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


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_all_orbs2(int[:, :] where, int[:, :] site_ranges):
    cdef int[:, :] offsets = np.empty((where.shape[0], 2), dtype=gint_dtype)
    cdef int[:, :] norbs = np.empty((where.shape[0], 2), dtype=gint_dtype)

    cdef int w, a, a_offset, a_norbs, b, b_offset, b_norbs
    for w in range(where.shape[0]):
        a = where[w, 0]
        _get_orbs(site_ranges, a, &a_offset, &a_norbs)
        if where.shape[1] == 1:
            b, b_offset, b_norbs = a, a_offset, a_norbs
        else:
            b = where[w, 1]
            _get_orbs(site_ranges, b, &b_offset, &b_norbs)
        offsets[w, 0] = a_offset
        offsets[w, 1] = b_offset
        norbs[w, 0] = a_norbs
        norbs[w, 1] = b_norbs

    return offsets, norbs


def _get_tot_norbs(syst):
    cdef int _unused, tot_norbs
    is_infinite_system = isinstance(syst, InfiniteSystem)
    n_sites = syst.cell_size if is_infinite_system else syst.graph.num_nodes
    _get_orbs(np.asarray(syst.site_ranges, dtype=gint_dtype),
              n_sites, &tot_norbs, &_unused)
    return tot_norbs


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

cdef int _check_onsite2(complex[:, :] M, int norbs,
                       int check_hermiticity) except -1:
    "Check onsite matrix for correct shape and hermiticity."
    if M.shape[0] != M.shape[1]:
        raise UserCodeError('Onsite matrix is not square')
    if M.shape[0] != norbs:
        raise UserCodeError(_shape_msg.format('Onsite'))
    if check_hermiticity and not _is_herm_conj2(M, M):
        raise ValueError(_herm_msg.format('Onsite'))
    return 0
