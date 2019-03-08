
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
import tkwant
# cimport kwant.operator
# from kwant.operator cimport _LocalOperator
from kwant.graph.defs cimport gint
# from kwant cimport operator
from kwant.graph.defs import gint_dtype
#from kwant.system import InfiniteSystem
###FOR _check_ham
from kwant._common import UserCodeError, get_parameters
import AdelsIdea.generalOperatorPathsbefore


import numpy as np
# from libcpp.vector cimport vector
# gint_dtype = np.int32


def zerofct(*args, **kwargs):
    return 0

def add_two_lead_unit_cells(syst, lead, phase_func=zerofct):
    first_leaduc = tkwant.leads.add_voltage(syst, lead, phase_func)
    scnd_leaduc = tkwant.leads.add_voltage(syst, lead, zerofct)
    return first_leaduc, scnd_leaduc

class heatCurrentWithIc(kwant.operator._LocalOperator):
    r"""
    The heat current is given by I_h = I_E - mu * I_N + I_c,
    where I_E is the energy current, I_N is the particle current and
    I_c is an additional term (coupling) which is needed.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator._LocalOperator.__call__` for details.

   Returns: :math:`0.5 * I_E - mu * I_N - 0.5 * \partial H/\partial t G_{0m} - 0.5 * I^E_{shifted}`
   for a given scattering state, i.e. for a given time, energy and alpha(=lead
   of scat state). Here, we use the explicit term of I_c which is
   :math:`I_c = -0.5 * I_E - 0.5 * \partial H/\partial t G_{0m} - 0.5 * I^E_{shifted}`.

   Parameters
   ----------
   syst : `~kwant.system.System`
   mu : chemical potential of the lead under investigation
   intracell_sites: A list of sites of the 1st unit cell of the lead.
       Either as `~kwant.builder.Site` or as Site of the finalized system
       (i.e. of type int).
   intercell_sites: A list of sites of the 2nd unit cell of the lead.
       Either as `~kwant.builder.Site` or as Site of the finalized system
       (i.e. of type int).
   tderiv_Hamil : the time derivative of the Hamiltonian. If unspecified, a
                  finite difference is used to approximate d/dt H
   del_t_deriv : small time-step used to calculate the difference quotient of H(t)

    """

    @cython.embedsignature
    def __init__(self, syst, mu, intracell_sites, intercell_sites, *,
                 del_t_deriv=1e-3, tderiv_Hamil=None):
        r"""
        Initialize the different terms needed:
        EnergyCurrent, ParticleCurrent, explicite t-dep term and shifted I^E
        """
        self.mu = mu
        #Create instances of EnergyCurrent and Particle Current
        self.energyCurrent = LeadEnergyCurrent(syst, intracell_sites, intercell_sites, check_hermiticity=True)

        curr_where = self.energyCurrent.get_onwhere()
        self.particleCurrent = AdelsIdea.generalOperatorPathsbefore.Current(syst, onsite=1, where=curr_where, \
                 check_hermiticity=True, sum=True)

        # derivative of the Hamiltonian by finite difference
        def diff_Hamil(a,b,*args, **kwargs):
            time = args[0]
            retfunc = (syst.hamiltonian(a, b, time+del_t_deriv, *args[1:], **kwargs) \
                    - syst.hamiltonian(a, b, time, *args[1:], **kwargs))/del_t_deriv
            return retfunc
        # use d/dt H above if not otherwise specified
        if tderiv_Hamil == None:
            Hdot = diff_Hamil
        else:
            Hdot = tderiv_Hamil
        #Create instance of explicitely t-dep terms
        self.tdepCoupling = AdelsIdea.generalOperatorPathsbefore.ArbitHop(syst, onsite=1, \
                            arbit_hop_func=Hdot, where=curr_where, \
                            check_hermiticity=False, sum=True)

        #Create instance of of I^E_{shifted}
            # Create needed site-lists
        sitesAttachedToLead = _create_list_of_certain_neighbors(syst,
                                        intracell_sites, intercell_sites)
        neighSitesAttachedToLead = _create_list_of_certain_neighbors(syst,
                                        sitesAttachedToLead, intracell_sites)

        self.energyCurrentShifted = LeadEnergyCurrent(syst, sitesAttachedToLead,
                        neighSitesAttachedToLead, check_hermiticity=True)


    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        """
        Calculate the initialized operators given bra (and ket).
        """

        return_Ecurr = self.energyCurrent(bra, args = args, params=params)
        return_Ncurr = self.particleCurrent(bra, args = args, params=params)
        return_Ecurr_shift = self.energyCurrentShifted(bra, args = args, params=params)
        return_expl_time_dep = self.tdepCoupling(bra, args = args, params=params)
        if return_expl_time_dep.imag:
            assert(abs(return_expl_time_dep.imag) < 1e-14)
        return_expl_time_dep_real = return_expl_time_dep.real


        return - 0.5 * return_Ecurr + self.mu * return_Ncurr +\
               + 0.5 *  (return_Ecurr_shift - return_expl_time_dep_real)



class heatCurrentNoIc(kwant.operator._LocalOperator):
    """ The heatCurrent is given by I_h = I_E - mu * I_N + I_c, where I_E is the energy current, I_N is the particle current and I_c is an additional term which might be needed from a physical point of view, but is set to 0 in this class.

   Returns:  I_E - mu * I_N for a given scattering state, i.e. for a given time, energy and alpha(=lead and mode of scattering state).
    """

    @cython.embedsignature
    def __init__(self, syst, mu, intracell_sites, intercell_sites, *, check_hermiticity=True):
        """Initialize the different terms needed: EnergyCurrent and ParticleCurrent"""

        self.mu = mu
        #Create instances of EnergyCurrent and Particle Current
        self.energyCurrent = LeadEnergyCurrent(syst, intracell_sites, intercell_sites, check_hermiticity=True)

        curr_where = self.energyCurrent.get_onwhere()
        self.particleCurrent = AdelsIdea.generalOperatorPathsbefore.Current(syst, onsite=1, where=curr_where,
                 check_hermiticity=True, sum=True)


    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        """Calculate particle and energy current for a given bra (and ket). Like this, the scattering wave function in kwant has to be calculated only once per E and t."""

        return_Ecurr = self.energyCurrent(bra, args = args, params=params)
        return_Ncurr = self.particleCurrent(bra, args = args, params=params)

        return return_Ecurr - self.mu * return_Ncurr



class LeadEnergyCurrent(kwant.operator._LocalOperator):
    r"""An operator for calculating the energy current of into/from(?) a Lead.

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
    intracell_sites: A list of sites of the 1st unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    intercell_sites: A list of sites of the 2nd unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.

    Notes
    -----
    We want to calculate the time-dependent energy current in analogy to the
    particle current which is given in tight-binding representation between
    site  :math:`i` and  :math:`j` by
      :math:`I_{ij}^N = -2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
      Im\left[ (\psi_i^{\alpha E})^\dagger(t)` H_{ij} \psi_j^{\alpha E}(t) \right]`,
    with the scattering states :math:`\psi^{\alpha E}_i(t)` from lead :math:`\alpha`
    at energy :math:`E` and time :math:`t`.
    For the current in the lead, one has to some over the cross section.

    The formula to calculate the energy current in a lead in tight-binding
    representation only changes the term in the energy integral of which the
    imaginary part is taken. It reads:
    Im\left[ \sum_{i,j \in \text{lead}} \sum_{q\in \bar{0}} (\psi_q^{\alpha E})^\dagger(t)` H_{qi} H_{ij} \psi_j^{\alpha E}(t) \right]`,
    where the sum over i,j runs over all sites in the corresponding lead and q
    runs over all sites in the scattering region :math:`\bar{0}`.
    Compared to the current, an additional Hamiltonian term is needed for the
    energy current.
    The sum over the lead is devided into two parts:
    The 'onSite' part, i.e. :math:`i==j` :
    Im\left[ \sum_{i \in \text{lead}} \sum_{q\in \bar{0}} (\psi_q^{\alpha E})^\dagger(t)` H_{qi} H_{ii} \psi_i^{\alpha E}(t) \right]`,
    which can be calculated with `~kwant.operator.Current` with an Onsite-term
    that is the Hamiltonian itself.
    The 'offSite' part, i.e. :math:`i\neq j`:
    Im\left[ \sum_{i\neq j \in \text{lead}} \sum_{q\in \bar{0}} (\psi_q^{\alpha E})^\dagger(t)` H_{qi} H_{ij} \psi_j^{\alpha E}(t) \right]`,
    which is calculated by the operator 'offEnergyCurrent' defined here.
    """
    @cython.embedsignature
    def __init__(self, syst, intracell_sites, intercell_sites, *, check_hermiticity=True):

        #check if site lists are a list of Sites or Integers(=finalized Sites) and make it list of Integers
        if isinstance(intracell_sites[0], kwant.builder.Site):
            intracell_sites_final = list(syst.id_by_site[s] for s in intracell_sites)
        else:
            assert(isinstance(intracell_sites[0],int))
            intracell_sites_final = intracell_sites
        if isinstance(intercell_sites[0], kwant.builder.Site):
            intercell_sites_final = list(syst.id_by_site[s] for s in intercell_sites)
        else:
            assert(isinstance(intercell_sites[0],int))
            intercell_sites_final = intercell_sites

        cdef gint[:,:] relPathList
        #where-lists creation
        onwhere, offwhere, relPathList = \
                    _create_where_and_path_lists_from_added_sites_for_lead(syst,
                                        intracell_sites=intracell_sites_final,  intercell_sites=intercell_sites_final)

        #create again `~kwant.builder.Site`-lists because of the method
        #'kwant.operator._normalize_hopping_where' which is called later
        self.onwhere = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in onwhere)
        self.offwhere = list([(syst.sites[hop[0]],syst.sites[hop[1]]) for hop in list] for list in offwhere)

        #initialize 'offSite' term of Energy Current
        self.offSite = AdelsIdea.generalOperatorPathsbefore.offEnergyCurrent(
                                syst, self.offwhere, relPathList=relPathList,
                                check_hermiticity=check_hermiticity, sum=True)

        #initialize 'onSite' term of Energy Current
        #and with the matching onSite-Hamiltonian function
        def onsiteHamil(a, *args, **params):
            if type(a) == kwant.builder.Site:
                a = syst.id_by_site[a]
            assert(type(a) == int)
            return syst.hamiltonian(a, a, *args, params=params)
        self.onSite = AdelsIdea.generalOperatorPathsbefore.Current(syst, onsite=onsiteHamil, where=self.onwhere, check_hermiticity=check_hermiticity, sum=True)

    def get_onwhere(self):
        return self.onwhere

    def get_offwhere(self):
        return self.offwhere


    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        r"""Calculate the energy current of a lead for both, :math:`i==j` and :math:`i\neq j` parts ( :math:`i` and  :math:`i` in lead)

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
        `float` that is the sum of 'onSite' and 'offSite' parts for a given bra.
        """

        resultoff = self.offSite(bra, ket, args=args, params=params)
        resulton = self.onSite(bra, ket, args=args, params=params)

        # The following minus sign is needed to consider the correct directions
        # of the onSiteCurrent (alternatively change hop[0] <-> hop[1] in onwhere)
        return - resulton + resultoff


class LocalEnergyCurrent(kwant.operator._LocalOperator):
    r"""
    An operator for calculating the local energy current between two sites.

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
    intracell_sites: A list of sites of the 1st unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    intercell_sites: A list of sites of the 2nd unit cell of the lead.
        Either as `~kwant.builder.Site` or as Site of the finalized system (i.e. of type int).
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.

    Notes
    -----
    We want to calculate the time-dependent energy current in analogy to the
    particle current which is given in tight-binding representation between
    site  :math:`i` and  :math:`j` by
      :math:`I_{ij}^N = -2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
      Im\left[ (\psi_i^{\alpha E})^\dagger(t)` H_{ij} \psi_j^{\alpha E}(t) \right]`,
    with the scattering states :math:`\psi^{\alpha E}_i(t)` from lead :math:`\alpha`
    at energy :math:`E` and time :math:`t`.
    For the current in the lead, one has to some over the cross section.

    The formula to calculate the local energy currentin tight-binding
    representation only changes the term in the energy integral and reads:
    :math:`I_{ij}^E = -2\sum^\text{leads}_\alpha\int \frac{dE}{2\pi} f_\alpha(E)
     \sum_{k} 0.5* Im\left[ (\psi_k^{\alpha E})^\dagger(t)` H_{ki} H_{ij} \psi_j^{\alpha E}(t) - (\psi_k^{\alpha E})^\dagger(t)` H_{kj} H_{ji} \psi_i^{\alpha E}(t) \right]`,
    where i and j are the given sites where the local energy current is to be
    calculated and k runs formally over all sites in the whole system (effectively
    only k are all neighbors of i and j, respectively).
    Compared to the current, an additional Hamiltonian term is needed for the
    heat current.
    In total, we divide the imaginary part above into 4 terms. In the first term,
    k runs over all neighbors of i, whereas in the second term, k runs over all
    neighbors of j. Moreover, we distinguish for technical reasons the case
    `i==k` and `j==k`, which are calculated by the instances of onEnergyCurrent
    and all other cases by instances of offEnergyCurrent.
    """



    @cython.embedsignature
    def __init__(self, syst, where=None, *, check_hermiticity=True, sum=False):

        self.check_hermiticity = check_hermiticity
        where_normalized = kwant.operator._normalize_hopping_where(syst, where)

        # extended where-lists creation and bookkeeping auxiliary lists
        offwhere_i, self.relPathList_i = \
            _create_where_and_path_lists_for_local_ECurr(syst, where_normalized)

        # kwant.operator.normalize wants unfinalized sites (BUG?)
        offwhere_i_unfnlzd = list([(syst.sites[hop[0]],syst.sites[hop[1]])
                                                    for hop in sublist]
                                                    for sublist in offwhere_i)
        # initialize 'offSite' term of Energy Current
        self.offSite_i = AdelsIdea.generalOperatorPathsbefore.offEnergyCurrent(
                            syst, offwhere_i_unfnlzd, relPathList=self.relPathList_i,
                            check_hermiticity=check_hermiticity, sum=sum)

        # initialize 'onSite' term of Energy Current
        # with the onSite-Hamiltonian
        def onsiteHamil(a, *args, **params):
            if type(a) == kwant.builder.Site:
                a = syst.id_by_site[a]
            assert(type(a) == int)
            return syst.hamiltonian(a, a, *args, params=params)
        self.onSite_i = AdelsIdea.generalOperatorPathsbefore.Current(syst,
                                onsite=onsiteHamil, where=where,
                                check_hermiticity=check_hermiticity, sum=sum)
        # the same for i and j in where swapped (to ensure I_E^ij == -I_E^ji,
        # in case of time-independent hopping ij)
        where_norm_swapped = [(hop[1],hop[0]) for hop in where_normalized]
        # extended where-lists creation and bookkeeping auxiliary lists
        offwhere_j, self.relPathList_j = \
            _create_where_and_path_lists_for_local_ECurr(syst, where_norm_swapped)

        #initialize 'offSite' term of Energy Current
        offwhere_j_unfnlzd = list([(syst.sites[hop[0]],syst.sites[hop[1]])
                                                    for hop in sublist]
                                                    for sublist in offwhere_j)
        self.offSite_j = AdelsIdea.generalOperatorPathsbefore.offEnergyCurrent(
                            syst, offwhere_j_unfnlzd, relPathList=self.relPathList_j,
                            check_hermiticity=check_hermiticity, sum=sum)

        where_unfinalized = list((syst.sites[hop[0]],syst.sites[hop[1]]) for hop in where_norm_swapped)
        #initialize 'onSite' term of Energy Current
        self.onSite_j = AdelsIdea.generalOperatorPathsbefore.Current(syst, onsite=onsiteHamil, where=where_unfinalized, check_hermiticity=check_hermiticity, sum=sum)



    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        r"""Calculate the local energy current for both, :math:`i==j` and
        :math:`i\neq j` parts ( :math:`i` and  :math:`i` in lead)

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
        `float` that is the sum of 'onSite' and 'offSite' parts for a given bra.
        """

        cdef gint i, tmp_hop_pos
        cdef gint j = 0
        cdef complex tmpsum = 0
        cdef complex res

        take_realpart = (self.check_hermiticity and (bra is ket or ket==None))

        resulton_i = self.onSite_i(bra, ket, args=args, params=params)
        resulton_j = self.onSite_j(bra, ket, args=args, params=params)
        assert len(resulton_i) == len(resulton_j)

        resultoff_i = self.offSite_i(bra, ket, args=args, params=params)
        # sum over all neighbors (k)
        assert len(self.relPathList_i) == len(resultoff_i)
        tmp_hop_pos = self.relPathList_i[0][0]
        for i, res in enumerate(resultoff_i):
            if self.relPathList_i[i][0] == tmp_hop_pos:
                tmpsum += res
            else:
                if take_realpart:
                    resultoff_i[j] = tmpsum.real
                else:
                    resultoff_i[j] = tmpsum
                j += 1
                tmp_hop_pos = self.relPathList_i[i][0]
                tmpsum = res

        if take_realpart:
            resultoff_i[j] = tmpsum.real
        else:
            resultoff_i[j] = tmpsum
        j += 1
        assert j == len(resulton_i)


        resultoff_j = self.offSite_j(bra, ket, args=args, params=params)
        # sum over all neighbors (k)
        assert len(self.relPathList_j) == len(resultoff_j)
        tmp_hop_pos = self.relPathList_j[0][0]
        j=0
        tmpsum = 0
        for i, res in enumerate(resultoff_j):
            if self.relPathList_j[i][0] == tmp_hop_pos:
                tmpsum += res
            else:
                if take_realpart:
                    resultoff_j[j] = tmpsum.real
                else:
                    resultoff_j[j] = tmpsum
                j += 1
                tmp_hop_pos = self.relPathList_j[i][0]
                tmpsum = res

        if take_realpart:
            resultoff_j[j] = tmpsum.real
        else:
            resultoff_j[j] = tmpsum
        j += 1
        assert j == len(resulton_j)

        resultoff_i = np.resize(resultoff_i,len(resulton_i))
        resultoff_j = np.resize(resultoff_j,len(resulton_j))

        if take_realpart:
            resultoff_i = resultoff_i.real
            resultoff_j = resultoff_j.real

        result = 0.5 * (-resulton_j + resultoff_j + resulton_i - resultoff_i)

        return result




def _create_where_and_path_lists_from_added_sites_for_lead(fsyst, intracell_sites, intercell_sites):
    r"""Creates where lists from the sitelists 'intracell_sites' and 'intercell_sites' for lead energy current calculation.

    Parameters
    ----------
    intracell_sites: list of all sites of finalized system in 1st lead unit cell
                     (i.e. of type 'int', e.g. by fsyst.id_by_site[sites])

    intercell_sites: list of all sites of finalized system (i.e. of type 'int')
                     in 2nd lead unit cell

    Returns
    -------
    lists with needed hoppings (tupels of 'int') or indexes where to find the
    needed hoppings:
    lead_scatreg_where: hoppings from scatregion to lead

    where: all needed hoppings for lead energy current

    path_list: list of all connected paths, more precisely the relative
               positions of the connected hoppings in the different wheres
    """

    where = []
    lead_scatreg_where = []

    #auxlists to store hoppings
    central_whereaux = []
    lead_whereaux = []

    path_list = []

    cdef gint lead_offset = 0
    cdef gint scat_offset = 0

    cdef gint i_idx, i, num_leadscat_hops, num_leadlead_hops
    # fill neighborlists; contains on purpose empty lists, if there is
    # no matching neighbor in the corresp. region
    for i_idx, i in enumerate(set(intracell_sites)):
        assert(type(i) == int)
        central_whereaux.append([])
        lead_whereaux.append([])
        for iedge in fsyst.graph.out_edge_ids(i):
            neighbor = fsyst.graph.head(iedge)
            #neighbor in lead
            if neighbor in set(intercell_sites+intracell_sites):
                lead_whereaux[i_idx].append( (i, neighbor) )
            #neighbor in scattering region
            else:
                central_whereaux[i_idx].append( (neighbor, i) )

        # Storing the connected paths in terms of relative positions in the
        # where lists central-lead-hopping (iq) and lead-lead-hoppings (ij)
        # which have the same lead site 'i'
        for num_leadscat_hops in range(len(central_whereaux[i_idx])):
            for num_leadlead_hops in range(len(lead_whereaux[i_idx])):
                path_list.append( (scat_offset+num_leadscat_hops,
                                   lead_offset+num_leadlead_hops) )
                # wherepos_neigh_stacked.append(wherepos_neigh_dummy)
        scat_offset += len(central_whereaux[i_idx])
        lead_offset += len(lead_whereaux[i_idx])


    # get seperate and total where
    lead_scatreg_where = [tupel for iSiteHoppings in central_whereaux for tupel in iSiteHoppings]
    lead_lead_where = [tupel for iSiteHoppings in lead_whereaux for tupel in iSiteHoppings]
    where.append(lead_scatreg_where)
    where.append(lead_lead_where)

    return lead_scatreg_where, where, np.asarray(path_list, dtype=gint_dtype)



def _create_where_and_path_lists_for_local_ECurr(fsyst, in_where):
    r"""
    Creates where lists from the hopping-list 'in_where' for local energy current calculation.

    Parameters
    ----------
    fsyst: finalized system under consideration

    in_where: list of all hoppings where the local energy current is to be
              calculated of finalized system (i.e. of type 'int')

    Returns
    -------
    lists with needed (additional) hoppings (tupels of 'int') and auxiliary lists for bookkeeping (where to find the needed hoppings):
    fullwhere: flattened list of all needed hoppings (not only ij, but also ik or jk)

    path_list: list of all connected paths, more precisely the relative
               positions of the connected hoppings in the different wheres
    """

    neigh_whereaux = []
    path_list = []

    neigh_count = 0

    # list to be extended
    fullwhere = [(hop[0],hop[1]) for hop in in_where]

    # find needed neighbors and store additional hoppings
    for i_idx, hop in enumerate(in_where):
        i = hop[1]
        assert(type(i) == int or type(i) == np.int32)

        start_neigh_count = neigh_count
        # find neighbors of hop[1]
        for iedge in fsyst.graph.out_edge_ids(i):
            neighbor = fsyst.graph.head(iedge)
            # store new hopping
            neigh_whereaux.append((i,neighbor))
            # how many hoppings have been added
            path_list.append( (i_idx, neigh_count) )
            neigh_count += 1

    # append new hoppings to where
    fullwhere = [fullwhere]
    fullwhere.append(neigh_whereaux)

    return fullwhere, path_list





def _create_list_of_certain_neighbors(fsyst, initial_list, forbidden_list):
    r"""
    Creates a list of sites, which are neighbors the sites in 'initial_list'
    but which are neither in 'forbidden_list' nor in 'initial_list'.
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

    # create list in which the neighbors of 'initial_list' which are not in
    # 'forbidden_list' nor in 'initial_list' are stored.
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
