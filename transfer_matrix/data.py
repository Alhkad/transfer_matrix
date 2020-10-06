from os.path import join 

def load_data(path):
    """
    Loads parameters of layered structure.
    
    Input file should contain parameters for each layer in the following format:
            epsilon_1 thickness_1
            ...
            epsilon_n thickness_n
            
    Thickness is expected to be given in mkm.
    If the first or the last layer is semiinfinite thickness should be equal to zero.
    """
    
    epsilons = []
    thicknesses = []
    with open(join(path)) as file:
        for line in file:
            line = line.split('#')[0].strip()
            if not line:
                continue
            eps, d = line.split()
            epsilons.append(complex(eps))
            thicknesses.append(float(d))
    return epsilons, thicknesses