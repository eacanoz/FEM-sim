# Class Material -> Store material properties


class material:

    def __init__(self, name, **kwargs):

        """

        Parameters
        ------------

        name:       Name of material

        ++kwargs:

        k       :       Thermal conductivity [W m-1 K-1]
        miu     :       Dynamic viscosity [kg s-1 m-1]
        rho     :       Density [kg m-3]
        ...

        """

        self.name = name
        self.__dict__.update(kwargs)



