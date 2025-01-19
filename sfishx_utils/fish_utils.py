import os
import sys
import numpy as np
from getdist import plots, loadMCSamples
from getdist.gaussian_mixtures import GaussianND
import getdist
from cosette.varie import latex_pnames
import flint

class FishUtils:
    """ 
    Utilities to read the fisher matrices

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

    
    
    def get_euclid(self, spec, npt, dr1fsky, fix_tau = False, file_name = None):
        """ get Euclid alone fisher matrices

            Args:
                spec (string): Euclid specifications (opti or pess)
        """

        cosmo_model = set_model(self.model)

        return euclid_alone(self.fish_folder_euclid_alone, self.stephane, self.steph_model, spec, cosmo_model, npt, dr1fsky, fix_tau = fix_tau, file_name = file_name)


    def get_euclid_cmb(self, spec, cmb, mode, npt, dr1fsky, pr4, probes=None, fix_tau = False, file_name = None):
        """
        Get Euclid combined with CMB fisher matrices.
        
        Args:
            spec (str): Euclid specifications (opti or pess).
            cmb (str): CMB experiment (planck, SO or S4).
            mode (str): Way of combining CMB with Euclid (p, a or c):
                - "p" ("plus"): CMB + Euclid as independent probes.
                - "a" ("add"): Proper CMBxEuclid covariance but no cross-probes in data.
                - "c" ("cross"): Full CMBxEucld covariance and data.
            probes (str, optional): CMB probes to consider. Options are None or 'phionly'. 
                If None, it uses all CMB probes (TTTEEElens). If 'phionly', it considers only CMBlensing.
            fix_tau (bool, optional): Whether to fix the optical depth parameter tau in the fishers.
            file_name (str, optional): Custom file name for the fisher matrix.
        """

        if probes == 'phionly':
            fix_tau = False

        cosmo_model = set_model(self.model)
        return euclid_cmb(self.fish_folder, self.stephane, self.steph_model, spec, cmb, mode, cosmo_model, npt, dr1fsky, pr4, probes, fix_tau = fix_tau, file_name = file_name)


    def get_cmb(self, cmb, npt, probes=None, pr4=False, file_name = None):
        """ get CMB alone fisher matrices

            Args:
                cmb (string): CMB experiment (planck, SO or S4)
                probes (string, defaults to None): CMB probes to consider. 
                                                   Options are None or 'primonly'. If None it uses
                                                   all CMB probes (TTTEEElens), if 'primonly' it 
                                                   consider only CMB primaries(TTTEEE)
        """

        cosmo_model = set_model(self.model)
        return cmb_alone(self.fish_folder, self.stephane, self.steph_model, cmb, cosmo_model, npt, probes, pr4, file_name = file_name)


def set_model(model):
    """
    Sets a dictionary of model parameters based on the input model string. The dictionary includes the model name and its LaTeX representation.
    """
    model_dict = {'model' : model}

    match model.lower():
        case s if s.startswith('lcdm'):
            model_dict['model_tex'] = r'$\Lambda \rm CDM'
        case s if s.startswith('ig'):
            model_dict['model_tex'] = r'$\rm IG'
        case s if s.startswith('nmcxpdelta'):
            model_dict['model_tex'] = r'${\rm NMC}+ (\xi, \Delta)'
        case s if s.startswith('w0wa'):
            model_dict['model_tex'] = r'$w_0w_a$CDM'
        case _:
            model_dict['model_tex'] = r'Undefined'
    
    return model_dict


def euclid_alone(fish_folder, stephane, steph_model, spec, model, npt, dr1fsky, fix_tau = False, file_name = None):
    """
    Compute the Euclid-alone Fisher matrix.

    Args:
        fish_folder (str): Path to the folder containing the Fisher matrix files.
        stephane (bool): Whether to use the Stephane fishers.
        steph_model (str): The model name for the Stephane fishers..
        spec (str): The Euclid survey specification.
        model (str): The cosmological model.
        npt (str): The number of points used in the Fisher matrix.
        dr1fsky (bool): Whether to use the DR1 sky fraction.
        fix_tau (bool, optional): Whether to fix the optical depth parameter tau in the fishers.
        file_name (str, optional): Custom file name for the Fisher matrix.

    Returns:
        dict: A dictionary containing the Euclid-alone Fisher matrix.
    """

    if stephane:
        # Euclid_alone['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_" + steph_model + "_" + "max-bins_super-prec_21point.npz")
        filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_" + steph_model + "_" + "max-bins_super-prec_21point.npz")

    elif file_name is not None:
        filename = os.path.join(fish_folder, file_name)

    else:
        if dr1fsky:
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_flat_" + "max-bins_super-prec_dr1_fsky_" + npt + ".npz")
        else:
            filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_flat_" + "max-bins_super-prec_" + npt + ".npz")
    
    euclid_alone_dict = build_dict(filename, model, spec, fix_tau)
    euclid_alone_dict['label'] = "$3\times 2$pt Euclid " + spec,

    return euclid_alone_dict


def euclid_cmb(fish_folder, stephane, steph_model, spec, cmb, mode, model, npt, dr1fsky, pr4, probes = None, fix_tau = False, file_name = None):
    
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

    elif file_name is not None:
        filename = os.path.join(fish_folder, file_name)

    else:
        if probes == 'phionly':
            if (dr1fsky):
                if pr4 and cmb=='planck':
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "PR4_mode-" + mode + "_flat_max-bins_super-prec_dr1_fsky_" + npt + ".npz")
                else:
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_dr1_fsky_" + npt + ".npz")
            else:
                if pr4 and cmb=='planck':
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "PR4_mode-" + mode + "_flat_max-bins_super-prec_" + npt + ".npz")
                else:
                    filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_" + npt + ".npz")
        else:
            if pr4 and cmb=='planck':
                filename = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMB-" + cmb + "PR4_mode-" + mode + "_flat_max-bins_super-prec_" + npt + ".npz")
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


def cmb_alone(fish_folder, stephane, steph_model, cmb, model, npt, probes, pr4=False, file_name = None):

    label = 'CMB ' + cmb + '-like',

    if file_name is not None:
        filename = os.path.join(fish_folder, file_name)

    elif probes == 'primonly':
        label = "TTTEEE " + cmb + '-like'
        if stephane:
            filename = os.path.join(fish_folder, "fish_CMB-" + probes + "-" + cmb + "_" + steph_model + "_max-bins_super-prec_21point.npz")
        else:
            # filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "_flat_max-bins_super-prec_" + npt + ".npz")
            # da cambiare
            filename = os.path.join(fish_folder, "fish_CMB" + probes + "-" + cmb + "_flat_max-bins_super-prec_" + npt + ".npz")
    else:
        if stephane:
            filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "_" + steph_model + "_max-bins_super-prec_" + npt + ".npz")
        else:
            if cmb == 'planck' and pr4:
                filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "PR4_flat_max-bins_super-prec_" + npt + ".npz")
            else:
                filename = os.path.join(fish_folder, "fish_CMB-" + cmb + "_flat_max-bins_super-prec_" + npt + ".npz")

    cmb_alone = build_dict(filename, model)
    cmb_alone['label'] = label
    cmb_alone['cmb'] = cmb
    
    return cmb_alone


def build_dict(filename, model, spec = None, fix_tau = False):
    fish_dict = {
        'spec' : spec,
        'model' : model['model'],
        'model_tex': model['model_tex']
    }

    fish = np.load(filename, allow_pickle=True, encoding='latin1')

    fish_dict['fish_file'] = fish
    fish_dict['fish'] = fish['fish']
    fish_dict['cov'] = fish.get('cov')
    fish_dict['me'] = fish['me']
    fish_dict['wid'] = fish['wid']
    fish_dict['prec'] = fish['prec']
    fish_dict['all_params'] = fish['all_params']
    if b'om' in fish_dict['all_params']:
        fish_dict['all_params'] = ['om', 'ob', 'h', 'ns', 'sigma8', 'tau']
    for i, p in enumerate(fish_dict['all_params']):
        if p == 'Delta':
            fish_dict['all_params'][i] = 'delta_IG'

    print(fish_dict['all_params'])
    param_labels = latex_pnames(fish_dict['all_params'])
    print(param_labels)

    fish_dict['param_labels'] = param_labels

    fish_dict['fiducial'] = {}
    for i, p in enumerate(fish_dict['all_params']):
        fish_dict['fiducial'][p] = fish_dict['me'][i]

    if len(param_labels) != len(fish_dict['all_params']):
        print("Error, the lenght of param_labels is different form the length of param names\n"
              "You probably passed a fisher with a parameter not contemplated by the function latex_pnames")
        sys.exit()
    
    FMsample = GaussianND(fish_dict['me'], fish_dict['fish'],
                          is_inv_cov=True,
                          names = fish_dict['all_params'],
                          # labels = model['param_labels']
                          labels = param_labels
                          )
    fish_dict['FMsample'] = FMsample

    fish_dict['all_obs'] = fish['all_obs']
    fish_dict['ell_ranges'] = fish['ell_ranges']

    if fix_tau:
        return fix_tau_dict(fish_dict)
    else:
        return fish_dict

def fix_tau_dict(fish_dict):

    index_tau = np.where(fish_dict['all_params'] == 'tau')[0][0]
    fish_dict['all_params'] = np.delete(fish_dict['all_params'], index_tau)
    print(fish_dict['all_params'])

    fish_mat = np.delete(fish_dict['fish'], [index_tau], 0)
    fish_mat = np.delete(fish_mat, [index_tau], 1)
    fish_dict['fish'] = fish_mat

    cov_mat = np.linalg.inv(fish_mat)

    alt_fish = flint.arb_mat(fish_mat.shape[0], fish_mat.shape[1], fish_mat.flatten())
    alt_ifish = alt_fish.inv()
    flint_cov = np.array([float(ent) for ent in alt_ifish.entries()])
    flint_cov = flint_cov.reshape(fish_mat.shape[0], fish_mat.shape[1])

    fish_dict['cov'] = cov_mat

    wid = np.sqrt(np.diag(cov_mat))
    wid_flint = np.sqrt(np.diag(flint_cov))
    # assert np.allclose(wid, wid_flint)

    fish_dict['me'] = np.delete(fish_dict['me'], index_tau)
    fish_dict['wid'] = wid_flint

    prec = np.abs(fish_dict['wid'] / fish_dict['me'])
    for i, p in enumerate(fish_dict['all_params']):
        if fish_dict['me'][i] == 0.: # Special case to avoid infinite "prec"
            prec[i] = wid_flint[i]
    fish_dict['prec'] = prec

    param_labels = latex_pnames(fish_dict['all_params'])
    fish_dict['param_labels'] = param_labels
    print(fish_dict['param_labels'])

    fish_dict['fiducial'] = {}
    for i, p in enumerate(fish_dict['all_params']):
        fish_dict['fiducial'][p] = fish_dict['me'][i]

    if len(param_labels) != len(fish_dict['all_params']):
        print("Error, the lenght of param_labels is different form the length of param names\n"
              "You probably passed a fisher with a parameter not contemplated by the function latex_pnames")
        sys.exit()

    FMsample = GaussianND(fish_dict['me'], fish_dict['fish'],
                          is_inv_cov=True,
                          names = fish_dict['all_params'],
                          labels = fish_dict['param_labels']
                          )
    fish_dict['FMsample'] = FMsample

    return fish_dict