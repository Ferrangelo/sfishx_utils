import os
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
        fiducial,
        param_labels,
        steph_model = None,
        stephane = False,
        steph_tau = False
    ):

        self.model = model
        self.fiducial = fiducial
        self.param_labels = param_labels

        self.stephane = stephane
        self.steph_tau = steph_tau
        self.steph_model = steph_model

        if self.stephane:
            self.fish_folder = '/media/veracrypt1/work_archives/archives/Code/CMBXC/forecast/cmbx_forecasts_outputs/fishers/Stephane'
            if steph_tau:
                self.fish_folder_euclid_alone = '/media/veracrypt1/work_archives/archives/Code/CMBXC/forecast/cmbx_forecasts_outputs/fishers/Stephane/Euclid_with_tau'
            else:
                self.fish_folder_euclid_alone = self.fish_folder
        else:
            self.fish_folder_euclid_alone = folder
            self.fish_folder = folder
    
    
    def get_E(self, spec):
        """ get Euclid alone fisher matrices

            Args:
                spec (string): Euclid specifications (opti or pess)
        """

        cosmo_model = set_model(self.model, self.fiducial, self.param_labels)

        return euclid_alone(self.fish_folder_euclid_alone, self.stephane, self.steph_model, spec, cosmo_model)


    def get_Ecmb(self, spec, cmb, mode, probes=None):
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

        cosmo_model = set_model(self.model, self.fiducial, self.param_labels)
        return euclid_cmb(self.fish_folder, self.stephane, self.steph_model, spec, cmb, mode, cosmo_model, probes)

    def get_cmb(self, cmb, probes=None):
        """ get CMB alone fisher matrices

            Args:
                cmb (string): CMB experiment (planck, SO or S4)
                probes (string, defaults to None): CMB probes to consider. 
                                                   Options are None or 'primonly'. If None it uses
                                                   all CMB probes (TTTEEElens), if 'primonly' it 
                                                   consider only CMB primaries(TTTEEE)
        """

        cosmo_model = set_model(self.model, self.fiducial, self.param_labels)
        return cmb_alone(self.fish_folder, self.stephane, self.steph_model, cmb, cosmo_model, probes)


def set_model(model, fiducial, param_labels):
    model_dict = {'model' : model,
                  'fiducial' : fiducial,
                  'param_labels' : param_labels[:],
                }
    if model == 'LCDM':
        model_dict['model_tex'] = r'$\Lambda \rm CDM$'
    if model == 'IG':
        model_dict['model_tex'] = r'$\rm IG$'
    if model == 'NMCxpDelta':
        model_dict['model_tex'] = r'${\rm NMC}+ (\xi, \Delta)$'

    model_dict['param_names'] = [par for par in model_dict['fiducial']],
    model_dict['means'] = [model_dict['fiducial'][par] for par in model_dict['fiducial']],
    
    return model_dict

def euclid_alone(fish_folder, stephane, steph_model, spec, model):

    Euclid_alone = {
        'spec' : spec,
        'model' : model['model'],
        'label' : '3x2pt Euclid ' + spec,
        'model_tex': model['model_tex']
    }

    if stephane:
        Euclid_alone['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_" + steph_model + "_" + "max-bins_super-prec_21point.npz")
    else:
        Euclid_alone['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_flat_" + "max-bins_super-prec_21point.npz")

    fish = np.load(Euclid_alone['file'])
    Euclid_alone['fish_file'] = fish
    Euclid_alone['fish'] = fish['fish']
    Euclid_alone['cov'] = fish['cov']
    Euclid_alone['me'] = fish['me']
    Euclid_alone['wid'] = fish['wid']
    Euclid_alone['prec'] = fish['prec']
    Euclid_alone['param_names'] = model['param_names'][0]
    Euclid_alone['fiducial'] = model['fiducial']
    
    FMsample = GaussianND(model['means'][0], Euclid_alone['cov'],
                          #is_inv_cov=True,
                          names = model['param_names'][0],
                          labels = model['param_labels']
                          )
    Euclid_alone['FMsample'] = FMsample


    return Euclid_alone

def euclid_cmb(fish_folder, stephane, steph_model, spec, cmb, mode, model, probes = None):
    
    """
     mode means the way we combine the probes
    "_mode-a" or "mode-c" or "mode-p" == * p ("plus")  == CMB + Euclid as independent probes
                                         * a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
                                         * c ("cross") == full CMBxEucld covariance and data
    """

    Euclid_cmb = {
        'spec' : spec,
        'model' : model['model'],
        'cmb' : cmb,
        'mode' : mode,
        'model_tex': model['model_tex']
    }

    if stephane:
        if probes == 'phionly':
            Euclid_cmb['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_" + steph_model + "_max-bins_super-prec_21point.npz")
        else:
            Euclid_cmb['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMB-" + cmb + "_mode-" + mode + "_" + steph_model + "_max-bins_super-prec_21point.npz")
    else:
        if probes == 'phionly':
            Euclid_cmb['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMBphionly-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_21point.npz")
        else:
            Euclid_cmb['file'] = os.path.join(fish_folder, "fish_Euclid-" + spec + "_CMB-" + cmb + "_mode-" + mode + "_flat_max-bins_super-prec_21point.npz")

    if probes == 'phionly':
        if mode == 'p' or mode =='a':
            Euclid_cmb['label']= '3x2pt Euclid ' + spec + ' + CMBlens ' + cmb + '-like'
        else:
            Euclid_cmb['label']= '6x2pt Euclid ' + spec + ' ' + cmb + '-like'
    else:
        if mode == 'p' or mode =='a':
            Euclid_cmb['label']= '3x2pt Euclid ' + spec + ' + CMB ' + cmb + '-like'
        else:
            Euclid_cmb['label']= '3x2pt Euclid ' + spec + ' +X CMB ' + cmb + '-like'
    
    fish = np.load(Euclid_cmb['file'])
    Euclid_cmb['fish_file'] = fish
    Euclid_cmb['fish'] = fish['fish']
    Euclid_cmb['cov'] = fish['cov']
    Euclid_cmb['me'] = fish['me']
    Euclid_cmb['wid'] = fish['wid']
    Euclid_cmb['prec'] = fish['prec']
    Euclid_cmb['param_names'] = model['param_names'][0]
    Euclid_cmb['fiducial'] = model['fiducial']
    
    FMsample = GaussianND(model['means'][0], Euclid_cmb['cov'],
                          #is_inv_cov=True,
                          names = model['param_names'][0],
                          labels = model['param_labels']
                          )
    Euclid_cmb['FMsample'] = FMsample
   

    return Euclid_cmb


def cmb_alone(fish_folder, stephane, steph_model, cmb, model, probes):

    cmb_alone = {
        'cmb' : cmb,
        'model' : model['model'],
        'label' : 'CMB ' + cmb + '-like',
        'model_tex': model['model_tex']
    }
    
    if probes == 'primonly':
        cmb_alone['label'] = "TTTEEE " + cmb + '-like'
        if stephane:
            cmb_alone['file'] = os.path.join(fish_folder, "fish_CMB-" + probes + "-" + cmb + "_" + steph_model + "_max-bins_super-prec_21point.npz")
        else:
            cmb_alone['file'] = os.path.join(fish_folder, "fish_CMB-" + cmb + "_flat_max-bins_super-prec_21point.npz")
    else:
        if stephane:
            cmb_alone['file'] = os.path.join(fish_folder, "fish_CMB-" + cmb + "_" + steph_model + "_max-bins_super-prec_21point.npz")
        else:
            cmb_alone['file'] = os.path.join(fish_folder, "fish_CMB-" + cmb + "_flat_max-bins_super-prec_21point.npz")
    
    fish = np.load(cmb_alone['file'])
    cmb_alone['fish_file'] = fish
    cmb_alone['fish'] = fish['fish']
    cmb_alone['cov'] = fish['cov']
    cmb_alone['me'] = fish['me']
    cmb_alone['wid'] = fish['wid']
    cmb_alone['prec'] = fish['prec']
    cmb_alone['param_names'] = model['param_names'][0]
    cmb_alone['fiducial'] = model['fiducial']
    
    FMsample = GaussianND(model['means'][0], cmb_alone['cov'],
                          #is_inv_cov=True,
                          names = model['param_names'][0],
                          labels = model['param_labels']
                          )
    cmb_alone['FMsample'] = FMsample


    return cmb_alone