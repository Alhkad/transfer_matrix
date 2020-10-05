import numpy as np

from transfer_matrix.data import load_data
from transfer_matrix.routines import TransferMatrixRoutines, reflection_from_transfer_matrix
from transfer_matrix.utils import degrees_to_radians

class LayeredStructure:
    
    def __init__(self, path):
        epsilons, thicknesses = load_data(path)
        if len(thicknesses) != len(epsilons):
            # TODO: change the exception to a specific one
            raise Exception('The numbers of thicknesses and epsilons don\'t match') 
            
        self.transfer_matrix_routines = TransferMatrixRoutines(thicknesses, epsilons)
    
    def get_reflection_coefs_fixed_angle(self, angle, pol, lambd_min, lambd_max, N_points=1000):
        """
            Calculates reflection coefficient for wavelength in [lambd_min, lambd_max]
            for given angle and polarization

            angle: angel of incident light, degrees
            pol: polarization of incident light: S or P
            lambd_min: minimal wavelength, mkm
            lambd_max: maximal wavelength, mkm
            N_points: number of points in wavelength grid

            return: wavelengths, reflection_coefs
        """
        wavelengths = np.linspace(lambd_min, lambd_max, N_points)
        reflection_coefs = np.empty_like(wavelengths, dtype=np.complex128)
        theta = degrees_to_radians(angle)
        for idx, lambd in enumerate(wavelengths):
            tmat = self.transfer_matrix_routines.get_matrix(lambd, theta, pol)
            reflection_coefs[idx] = reflection_from_transfer_matrix(tmat)
        return wavelengths, reflection_coefs
    
    def get_reflection_coefs_fixed_wavelength(self, lambd, pol, angle_min=0, angle_max=89.9, N_points=400):
        """
            Calculates reflection coefficient for angles in [angle_min, angle_max]
            for given wavelength and polarization

            lambd: wavelength of incident light, mkm
            pol: polarization of incident light: S or P
            angle_min: minimal angle, degrees
            angle_max: maximal angle, degrees
            N_points: number of points in angles grid

            return: angles, reflection_coefs
        """
        angles = np.linspace(angle_min, angle_max, N_points)
        reflection_coefs = np.empty_like(angles, dtype=np.complex128)
        thetas = degrees_to_radians(angles)
        for idx, theta in enumerate(thetas):
            tmat = self.transfer_matrix_routines.get_matrix(lambd, theta, pol)
            reflection_coefs[idx] = reflection_from_transfer_matrix(tmat)
        return angles, reflection_coefs