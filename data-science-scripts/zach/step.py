x = np.linspace(-1, 1, 100)
def discontinuous1D(x):
    return np.heaviside(x, 1) * x**2

def discontinuous2D():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    return np.heaviside(y, 1) * y**2 - (
        np.heaviside(x, 1) * x**2 + (1 - np.heaviside(x,1) * -1)
    )
