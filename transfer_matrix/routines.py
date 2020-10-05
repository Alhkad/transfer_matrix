import numpy as np

def interface_function_for_pol(pol):
    """
    Returns suitable interface function for given polarization
    """
    if pol == 'S':
        return interface_s
    elif pol == 'P':
        return interface_p
    else:
        raise ValueError('wrong polarization')

def calc_kz(k0, kx, eps):
    return np.sqrt(np.square(k0) * eps - np.square(kx))

def interface_s(k0, kx, eps1, eps2):
    """
    Calculates tranfer matrix for given interface (case of S polarization)
    """
    def FS(k0, kx, eps):
        kz = calc_kz(k0, kx, eps)
        fs = kz / k0
        return np.matrix([[1, 1], [-fs, fs]])
    FS1 = FS(k0, kx, eps1)
    FS2 = FS(k0, kx, eps2)
    return np.linalg.inv(FS2) * FS1

def interface_p(k0, kx, eps1, eps2):
    """
    Calculates tranfer matrix for given interface (case of P polarization)
    """
    def FP(k0, kx, eps):
        kz = calc_kz(k0, kx, eps)
        fp = kz / (k0 * eps)
        return np.matrix([[fp, fp], [-1, 1]])
    FP1 = FP(k0, kx, eps1)
    FP2 = FP(k0, kx, eps2)
    return np.linalg.inv(FP2) * FP1

def propagate(k0, kx, eps, thickness):
    """
    Calculates tranfer matrix for light propagation in the layer
    """
    kz = calc_kz(k0, kx, eps)
    exp_left = np.exp(-1j * kz * thickness)
    exp_right = np.exp(1j * kz * thickness)

    return np.matrix([[exp_right, 0], [0, exp_left]])

# -------------------------------------------------------------------------

class TransferMatrixRoutines:
    
    def __init__(self, thicknesses, epsilons):
        self.thicknesses, self.epsilons = thicknesses, epsilons
        self.number_of_layers = len(self.thicknesses)
        
    def get_matrix(self, lambd, theta, pol):
        """
        Calculates tranfer matrix for whole structure for fixed wavelength, angle and polarization
        """
        interface = interface_function_for_pol(pol)

        k0 = 2 * np.pi / lambd
        kx = k0 * np.sin(theta) * np.sqrt(self.epsilons[0])
        tmat = np.matrix(np.eye(2, dtype=np.complex128))
        for i in range(self.number_of_layers - 1):
            tmat = propagate(k0, kx, self.epsilons[i], self.thicknesses[i]) * tmat
            tmat = interface(k0, kx, self.epsilons[i], self.epsilons[i + 1]) * tmat

        return tmat
    
def reflection_from_transfer_matrix(tmat):
    c, d = tmat[1, 0], tmat[1, 1]
    return -c / d