import os, sys
import numpy as np
from getdist import plots, loadMCSamples
from getdist.gaussian_mixtures import GaussianND
import getdist

class Fish_utils:

    """ Utilities to read the fisher matrices

        Args:
            folder: folder where to look for fishers, overridden if stephane = True
            model (string): String identifying the model (LCDM, IG, NMCxpDelta)
            fiducial (dict): name and values of parameters in fiducial cosmology
            param_labels (array): array with names of params in latex (e.g. r"\Omega_{\rm}")
            steph_model: name of the model in Stephan fisher that appears in the file name of the fishers
                        (flat_LCDM usually but options for w0wa also)
            stephane (bool, defaults to False): Use Stephane fishers or not
            stephane_tau (bool, defaults to False): Use Stephane Euclid fishers with tau or without tau
    """

    def __init__(
        self, 
        folder,
        model,
        steph_model = None,
        stephane = False,
        steph_tau = False
    ):

        self.model = model

        self.stephane = stephane
        self.steph_tau = steph_tau
        self.steph_model = steph_model

        if self.stephane:
            self.fish_folder = '/media/veracrypt1/work_archives/cmbx_forecasts_outputs/fishers/Stephane'
            if steph_tau:
                self.fish_folder_euclid_alone = '/media/veracrypt1/work_archives/cmbx_forecasts_outputs/fishers/Stephane/Euclid_with_tau'
            else:
                self.fish_folder_euclid_alone = self.fish_folder
        else:
            self.fish_folder_euclid_alone = folder
            self.fish_folder = folder
        self.fish_folder_euclid_alone = folder
        self.fish_folder = folder

    
    
    def get_E(self, spec, npt, dr1fsky):
        """ get Euclid alone fisher matrices

            Args:
                spec (string): Euclid specifications (opti or pess)
        """

        cosmo_model = set_model(self.model)

        return euclid_alone(self.fish_folder_euclid_alone, self.stephane, self.steph_model, spec, cosmo_model, npt, dr1fsky)


    def get_Ecmb(self, spec, cmb, mode, npt, dr1fsky, pr4, probes=None):
        """ get Euclid combined with CMB fisher matrices

            Args:
                spec (string): Euclid specifications (opti or pess)
                cmb (string): CMB experiment (planck, SO or S4)
                mode (string): way of combining CMB with euclid (p, a or c)
                               p ("plus")  == CMB + Euclid as independent probes
                               a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
                               c ("cross") == full CMBxEucld covariance and data
                probes (string, defaults to None): CMB probes to consider. 
                                                   Options are None or 'phionly'. If None it uses
                                                   all CMB probes (TTTEEElens), if 'phionly' it 
                                                   consider only CMBlensing
        """

        cosmo_model = set_model(self.model)
        return euclid_cmb(self.fish_folder, self.stephane, self.steph_model, spec, cmb, mode, cosmo_model, npt, dr1fsky, pr4, probes)


    def get_cmb(self, cmb, npt, probes=None):
        """ get CMB alone fisher matrices

            Args:
                cmb (string): CMB experiment (planck, SO or S4)
                probes (string, defaults to None): CMB probes to consider. 
                                                   Options are None or 'primonly'. If None it uses
                                                   all CMB probes (TTTEEElens), if 'primonly' it 
                                                   consider only CMB primaries(TTTEEE)
        """

        cosmo_model = set_model(self.model)
        return cmb_alone(self.fish_folder, self.stephane, self.steph_model, cmb, cosmo_model, npt, probes)


def set_model(model):
    model_dict = {'model' : model}

    if model.lower() == 'lcdm':
        model_dict['model_tex'] = r'$\Lambda \rm CDM$'
    if model.lower() == 'ig':
        model_dict['model_tex'] = r'$\rm IG$'
    if model.lower() == 'nmcxpdelta':
        model_dict['model_tex'] = r'${\rm NMC}+ (\xi, \Delta)$'
    
    return model_dict


def euclid_alone(fish_folder, stephane, steph_model, spec, model, npt, dr1fsky):

    if stephane:
        # Euclid_alone['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_" + steph_model + "_" + "max-bins_super-prec_21point.npz")
        filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_" + steph_model + "_" + "max-bins_super-prec_21point.npz")
    else:
        if dr1fsky:
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_flat_" + "max-bins_super-prec_dr1_fsky_" + npt + ".npz")
        else:
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_flat_" + "max-bins_super-prec_" + npt + ".npz")
    
    Euclid_alone = build_dict(filename, model, spec)
    Euclid_alone['label'] = '3x2pt Euclid ' + spec,

    return Euclid_alone


def euclid_cmb(fish_folder, stephane, steph_model, spec, cmb, mode, model, npt, dr1fsky, pr4, probes = None):
    
    """
     mode means the way we combine the probes
    "_mode-a" or "mode-c" or "mode-p" == * p ("plus")  == CMB + Euclid as independent probes
                                         * a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
                                         * c ("cross") == full CMBxEucld covariance and data
    """

    if stephane:
        if probes == 'phionly':
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_" + steph_model + "_max-bins_super-prec_21point.npz")
        else:
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMB-" + cmb + "_mode-" + mode + "_" + steph_model + "_max-bins_super-prec_" + npt + ".npz")
    else:
        if probes == 'phionly':
            if (dr1fsky):
                if pr4:
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "PR4_mode-" + mode + "_flat_max-bins_super-prec_dr1_fsky_" + npt + ".npz")
                else:
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_dr1_fsky_" + npt + ".npz")
            else:
                if pr4:
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "PR4_mode-" + mode + "_flat_max-bins_super-prec_" + npt + ".npz")
                else:
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_" + npt + ".npz")
        else:
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMB-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_" + npt + ".npz")

    if probes == 'phionly':
        if mode == 'p' or mode =='a':
            label = '3x2pt Euclid ' + spec + ' + CMBlens ' + cmb + '-like'
        else:
            label = '6x2pt Euclid ' + spec + ' ' + cmb + '-like'
    else:
        if mode == 'p' or mode =='a':
            label = '3x2pt Euclid ' + spec + ' + CMB ' + cmb + '-like'
        else:
            label = '3x2pt Euclid ' + spec + ' +X CMB ' + cmb + '-like'

    Euclid_cmb = build_dict(filename, model, spec)
    Euclid_cmb['label'] = label
    Euclid_cmb['cmb'] = cmb
    Euclid_cmb['mode'] : mode
    
    return Euclid_cmb


def cmb_alone(fish_folder, stephane, steph_model, cmb, model, npt, probes):

    label = 'CMB ' + cmb + '-like',

    if probes == 'primonly':
        label = "TTTEEE " + cmb + '-like'
        if stephane:
            filename = os.path.join(fish_folder, "fish_CMB-" + probes + "-" + cmb + "_" + steph_model + "_max-bins_super-prec_21point.npz")
        else:
            filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "_flat_max-bins_super-prec_" + npt + ".npz")
    else:
        if stephane:
            filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "_" + steph_model + "_max-bins_super-prec_" + npt + ".npz")
        else:
            filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "_flat_max-bins_super-prec_" + npt + ".npz")

    cmb_alone = build_dict(filename, model)
    cmb_alone['label'] = label
    cmb_alone['cmb'] = cmb
    
    return cmb_alone


def build_dict(filename, model, spec = None):
    fish_dict = {
        'spec' : spec,
        'model' : model['model'],
        'model_tex': model['model_tex']
    }

    fish = np.load(filename)

    fish_dict['fish_file'] = fish
    fish_dict['fish'] = fish['fish']
    fish_dict['cov'] = fish.get('cov')
    fish_dict['me'] = fish['me']
    fish_dict['wid'] = fish['wid']
    fish_dict['prec'] = fish['prec']
    fish_dict['param_names'] = fish['all_params']
    if b'om' in fish_dict['param_names']:
        fish_dict['param_names'] = ['om', 'ob', 'h', 'ns', 'sigma8', 'tau']

    param_labels = latex_pnames(fish_dict['param_names'], model)

    fish_dict['param_labels'] = param_labels

    fish_dict['fiducial'] = {}
    for i, p in enumerate(fish_dict['param_names']):
        fish_dict['fiducial'][p] = fish_dict['me'][i]

    if len(param_labels) != len(fish_dict['param_names']):
        print("Error, the lenght of param_labels is different form the length of param names\n"
              "You probably passed a fisher with a parameter not contemplated by the function latex_pnames")
        sys.exit()
    
    FMsample = GaussianND(fish_dict['me'], fish_dict['fish'],
                          is_inv_cov=True,
                          names = fish_dict['param_names'],
                          # labels = model['param_labels']
                          labels = param_labels
                          )
    fish_dict['FMsample'] = FMsample

    return fish_dict


def latex_pnames(pnames, model):
    latex_list = []
    for p in pnames:
        if (p == 'ob') or (p == b'ob'):
            latex_list.append(r"\Omega_{\rm b}")

        if (p == 'om') or (p == b'om'):
            latex_list.append(r"\Omega_{\rm m}")

        if (p == 'w0') or (p == b'w0'):
            latex_list.append(r"w_0")

        if (p == 'wa') or (p == b'wa'):
            latex_list.append(r"w_{\rm a}")

        if (p == 'sigma8') or (p == b'sigma8'):
            latex_list.append(r"\sigma_8")

        if (p == 'tau') or (p == b'tau'):
            latex_list.append(r"\tau")
            
        if (p == 'ns') or (p == b'ns'):
            latex_list.append(r"n_{\rm s}")

        if (p == 'h') or (p == b'h'):
            latex_list.append(r"h")

        if (p == 'delta_IG') or (p == 'Delta'):
            latex_list.append(r"\Delta")

        if p == 'gamma_IG':
            if model['model'].lower() == 'ig':
                latex_list.append(r"\gamma")
            else:
                latex_list.append(r"\xi")

        if p == 'mnu':
            latex_list.append(r"m_{\nu}")

        if p == 'aIA':
            latex_list.append(r"a_{\rm IA}")

        if p == 'eIA':
            latex_list.append(r"e_{\rm IA}")

        if p == 'bIA':
            latex_list.append(r"b_{\rm IA}")

        if p == 'b0':
            latex_list.append(r"b_{0}")

        if p == 'b1':
            latex_list.append(r"b_{1}")

        if p == 'b2':
            latex_list.append(r"b_{2}")

        if p == 'b3':
            latex_list.append(r"b_{3}")

        if p == 'b4':
            latex_list.append(r"b_{4}")

        if p == 'b5':
            latex_list.append(r"b_{5}")

        if p == 'b6':
            latex_list.append(r"b_{6}")

        if p == 'b7':
            latex_list.append(r"b_{7}")

        if p == 'b8':
            latex_list.append(r"b_{8}")

        if p == 'b9':
            latex_list.append(r"b_{9}")

        if p == 'b10':
            latex_list.append(r"b_{10}")

        if p == 'b11':
            latex_list.append(r"b_{11}")

        if p == 'b12':
            latex_list.append(r"b_{12}")

        if p == 'b13':
            latex_list.append(r"b_{13}")

    return latex_list