import numpy as np

class Fish_utils:
    def __init__(
        self,
        number=4,
    ):

        self.number = number

    def get_number(self):

        return self.number


def galaxy_distribution(z, zmean=0.9):
    """
    Galaxy distribution returns the function D(z) from the notes

    """
    z0 = zmean / np.sqrt(2.0)

    galaxy_dist = (z / z0) ** 2 * np.exp(-((z / z0) ** (1.5)))

    return galaxy_dist


def photo_z_distribution(
    z, zph, cb=1.0, zb=0, sb=0.05, c0=1.0, z0=0.1, s0=0.05, fout=0.1
):
    """
    Photo z distribution
    Eq. 115 and Tab. 5 of 1910.09273
    """

    return (1 - fout) / np.sqrt(2 * np.pi) / sb / (1 + z) * np.exp(
        -0.5 * (z - cb * zph - zb) ** 2 / (sb * (1 + z)) ** 2
    ) + fout / np.sqrt(2 * np.pi) / s0 / (1 + z) * np.exp(
        -0.5 * (z - c0 * zph - z0) ** 2 / (s0 * (1 + z)) ** 2
    )
