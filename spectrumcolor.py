from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SpectrumColor():
    """A class for conversion of spectral data into CIE XYZ, xyY, and Lab color space values."""
    
    def __init__(self, 
                 spec_wave, 
                 spec_int, 
                 use_illuminant: bool = True, 
                 is_absorbance: bool = False, 
                 observer_file: str = 'lin2012xyz2e_5_7sf.csv', 
                 illuminant_file: Optional[str] = 'illuminant_d65.csv'):
        """Constructor which requires spectrum wavelength and intensity data. To load directly from file instead, use SpectrumColor.from_file(...).

        Parameters
        ----------
        spec_wave : 
            Raw wavelength data
        spec_int : 
            Raw spectrum intensity data
        use_illuminant : bool, optional
            use standard illuminant for reflection/transmission data, by default True
        is_absorbance : bool, optional
            whether or not the data is absorbance for transmission data, by default False
        observer_file : str, optional
            file for standard observer file with columns (wavelength, x, y, z), wavelength in increasing order and evenly spaced, by default 'lin2012xyz2e_5_7sf.csv'
        illuminant_file : Optional[str], optional
            file for standard illuminant file with columns (wavelength, spectral power), wavelength in increasing order, by default 'illuminant_d65.csv'
        """
        self.spec_wave_raw = spec_wave
        self.spec_int_raw = spec_int
        self.use_illuminant = use_illuminant
        self.is_absorbance = is_absorbance
        # D65 2deg, used for XYZ/Lab conversion
        self._Xr = 95.047
        self._Yr = 100
        self._Zr = 108.883
        # Standard observer/color matching functions (wavelength, x, y, z)
        # Wavelength has even spacing, increasing order
        self.observer_file = observer_file
        # Standard illuminant for reflection/transmission (wavelength, spectral power)
        self.illuminant_file = illuminant_file
        self.load_observer(file=None)
        self.load_illuminant(file=None)
        self.spec_int = self.clean_spectrum(self.spec_wave_raw, self.spec_int_raw, self.is_absorbance)  
    
    @classmethod
    def from_file(cls, 
                  spec_file: str, 
                  use_illuminant: bool = True, 
                  is_absorbance: bool = False, 
                  int_col: str = 'UV-Vis Absorbance', 
                  wave_col: str = 'UV-Vis Wavelength',
                  observer_file: str = 'lin2012xyz2e_5_7sf.csv', 
                  illuminant_file: Optional[str] = 'illuminant_d65.csv'):
        """Alternate class method constructor to load spectrum data directly from file.

        Parameters
        ----------
        spec_file : str
            file for standard illuminant file with columns (wavelength, spectral power), wavelength in increasing order
        use_illuminant : bool, optional
            use standard illuminant for reflection/transmission data, by default True
        is_absorbance : bool, optional
            whether or not the data is absorbance for transmission data, by default False
        int_col : str, optional
            spectrum intensity column name (used when loading with pandas dataframe), by default 'UV-Vis Absorbance'
        wave_col : str, optional
            spectrum wavelength column name (used when loading with pandas dataframe), by default 'UV-Vis Wavelength'
        observer_file : str, optional
            file for standard observer file with columns (wavelength, x, y, z), wavelength in increasing order and evenly spaced, by default 'lin2012xyz2e_5_7sf.csv'
        illuminant_file : Optional[str], optional
            file for standard illuminant file with columns (wavelength, spectral power), wavelength in increasing order, by default 'illuminant_d65.csv'

        Returns
        -------
        SpectrumColor
            Returns an instance of SpectrumColor with spectrum data loaded from file
        """
        
        spec_wave_raw, spec_int_raw = cls.read_spectrum(spec_file, int_col, wave_col) 
        return cls(spec_wave_raw, spec_int_raw, use_illuminant, is_absorbance, observer_file, illuminant_file)
        
    def load_observer(self, file: Optional[str] = None):
        """Load standard observer color matching function data from file into attributes.

        Parameters
        ----------
        file : Optional[str], optional
            standard observer file, if None uses self.observer_file, by default None
        """
        if file is None:
            file = self.observer_file
        else:
            self.observer_file = file
        
        cmf = pd.read_csv(file, header=None)
        
        cmf.columns =['Wavelength', 'X', 'Y', 'Z']
        self.wave = cmf['Wavelength'].to_numpy()
        self.x_cmf = cmf['X'].to_numpy()
        self.y_cmf = cmf['Y'].to_numpy()
        self.z_cmf = cmf['Z'].to_numpy()
        self.del_wave = self.wave[1] - self.wave[0]
        
    def load_illuminant(self, file: Optional[str] = None):
        """Load standard illuminant data from file into attributes.

        Parameters
        ----------
        file : Optional[str], optional
            standard illuminant file, if None uses self.illuminant_file, by default None
        """
        if file is None:
            file = self.illuminant_file
        else:
            self.illuminant_file = file
        
        ill = pd.read_csv(file, header=None)
        
        ill.columns =['Wavelength', 'Ill']
        # interpolate onto observer wavelength
        self.ill = np.interp(self.wave, ill['Wavelength'], ill['Ill'])

    def load_spectrum(self, 
                      file: str, 
                      is_absorbance: bool = False, 
                      int_col: str = 'UV-Vis Absorbance', 
                      wave_col: str = 'UV-Vis Wavelength'):
        """Load spectrum data from file, clean spectrum data, and interpolate onto observer wavelengths.

        Parameters
        ----------
        file : Optional[str], optional
            Spectrum data file.
        is_absorbance : bool, optional
            whether or not the data is absorbance for transmission data, by default False
        int_col : str, optional
            spectrum intensity column name (used when loading with pandas dataframe), by default 'UV-Vis Absorbance'
        wave_col : str, optional
            spectrum wavelength column name (used when loading with pandas dataframe), by default 'UV-Vis Wavelength'
        """
        self.spec_wave_raw, self.spec_int_raw = self.read_spectrum(file, int_col, wave_col)        
       
        self.spec_int = self.clean_spectrum(self.spec_wave_raw, self.spec_int_raw, is_absorbance)
        
    @staticmethod            
    def read_spectrum(file, int_col: str, wave_col: str):
        """_summary_

        Parameters
        ----------
        file : Optional[str]
            Spectrum data file.
        int_col : str
            spectrum intensity column name (used when loading with pandas dataframe).
        wave_col : str
            spectrum wavelength column name (used when loading with pandas dataframe).

        Returns
        -------
        Tuple of wavelength array and intensity array
        """
        Abs = pd.read_csv(file, comment='#')
        
        spec_int = Abs[int_col].to_numpy()
        spec_wave = Abs[wave_col].to_numpy()
        return (spec_wave, spec_int)            
    
    def clean_spectrum(self, spec_wave, spec_int, is_absorbance: bool):
        """Returns cleaned spectrum intensity data without Nan/Inf or negative values. Data is then interpolated onto observer wavelengths.

        Parameters
        ----------
        spec_wave :
            Raw wavelength data
        spec_int :
            Raw spectrum intensity data
        is_absorbance : bool
            whether or not the data is absorbance for transmission data

        Returns
        -------
        Array containing cleaned spectrum intensity
        """
        mask1 = np.isnan(spec_int)
        mask2 = np.isinf(spec_int)
        mask = np.logical_or(mask1,mask2)

        # replace any Nan or Inf by interpolation
        spec_int[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spec_int[~mask])
        # interpolate onto observer wavelength
        spec_int = np.interp(self.wave, spec_wave, spec_int)
        # remove negative intensity
        spec_int[spec_int<0] = 0
        
        if is_absorbance:
            spec_int = 10**(-spec_int)
        
        return spec_int
    
    @property
    def Xr(self):
        return self._Xr
    
    @property
    def Yr(self):
        return self._Yr
    
    @property
    def Zr(self):
        return self._Zr
    
    @Xr.setter
    def Xr(self, val):
        if val < 5:
            print('Value must be in nominal range 0-100, not 0-1')
        else:
            self._Xr = val
            
    @Yr.setter
    def Yr(self, val):
        if val < 5:
            print('Value must be in nominal range 0-100, not 0-1')
        else:
            self._Yr = val
            
    @Zr.setter
    def Zr(self, val):
        if val < 5:
            print('Value must be in nominal range 0-100, not 0-1')
        else:
            self._Zr = val
    
    def calculate_XYZ(self) -> Tuple[float, float, float]:
        if self.use_illuminant:
            N = np.sum(self.y_cmf*self.ill*self.del_wave)
            X = np.sum(self.spec_int*self.x_cmf*self.ill*self.del_wave) / N * 100
            Y = np.sum(self.spec_int*self.y_cmf*self.ill*self.del_wave) / N * 100
            Z = np.sum(self.spec_int*self.z_cmf*self.ill*self.del_wave) / N * 100
        else:
            X = np.sum(self.spec_int*self.x_cmf*self.del_wave) * 100
            Y = np.sum(self.spec_int*self.y_cmf*self.del_wave) * 100
            Z = np.sum(self.spec_int*self.z_cmf*self.del_wave) * 100
        
        return (X, Y, Z)
    
    def calculate_Lab(self, Xr: Optional[float] = None, Yr: Optional[float] = None, Zr: Optional[float] = None) -> Tuple[float, float, float]:
        X, Y, Z = self.calculate_XYZ()
        return self.XYZ_to_Lab(X, Y, Z, Xr, Yr, Zr)
        
    def distance_to_Lab(self, L0: float, a0: float, b0: float, Xr: Optional[float] = None, Yr: Optional[float] = None, Zr: Optional[float] = None) -> float:
        L, a, b = self.calculate_Lab(Xr, Yr, Zr)
        distance = np.sqrt((L0-L)**2 + (a0-a)**2 + (b0-b)**2)
        return distance

    def calculate_xyY(self):
        X, Y, Z = self.calculate_XYZ()
        sum = X + Y + Z
        if sum == 0:
            # black
            X = self.Xr
            Y = self.Yr
            Z = self.Zr
            sum = X + Y + Z
            
        x = X / sum
        y = Y / sum
        
        return (x, y, Y)
        
    def Lab_to_XYZ(self, L: float, a: float, b: float, Xr: Optional[float] = None, Yr: Optional[float] = None, Zr: Optional[float] = None):
        if Xr is None:
            Xr = self.Xr
        if Yr is None:
            Yr = self.Yr
        if Zr is None:
            Zr = self.Zr
            
        eps = 0.008856
        kap = 903.3

        fy = (L + 16)/116
        fx = a/500 + fy
        fz = fy - b/200

        if fx**3 > eps:
            xr = fx**3
        else:
            xr = (116*fx - 16)/kap

        if L > kap*eps:
            yr = fy**3
        else:
            yr = L/kap
            
        if fz**3 > eps:
            zr = fz**3
        else:
            zr = (116*fz - 16)/kap

        X = xr*Xr
        Y = yr*Yr
        Z = zr*Zr
        
        return(X, Y, Z) # output from 0 to 100
    
    def XYZ_to_Lab(self, X: float, Y: float, Z: float, Xr: Optional[float] = None, Yr: Optional[float] = None, Zr: Optional[float] = None): 
        # input expected from 0 to 100
        if Xr is None:
            Xr = self.Xr
        if Yr is None:
            Yr = self.Yr
        if Zr is None:
            Zr = self.Zr
            
        eps = 0.008856
        kap = 903.3
        
        xr = X/Xr
        yr = Y/Yr
        zr = Z/Zr
        
        if xr > eps:
            fx = xr**(1/3)
        else:
            fx = (kap*xr + 16)/116
            
        if yr > eps:
            fy = yr**(1/3)
        else:
            fy = (kap*yr + 16)/116
            
        if zr > eps:
            fz = zr**(1/3)
        else:
            fz = (kap*zr + 16)/116
            
        L = 116*fy - 16
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        
        return (L, a, b)

    def plot_spectrum(self):
        plt.plot(self.wave, self.spec_int)
        plt.show()
    
    def plot_spectrum_raw(self):
        plt.plot(self.spec_wave_raw, self.spec_int_raw)
        plt.show()
        
    def plot_observer(self):
        plt.plot(self.wave, self.x_cmf, 'r', self.wave, self.y_cmf, 'g', self.wave, self.z_cmf, 'b')
        plt.show()
    
    def plot_illuminant(self):
        if self.illuminant is not None:
            plt.plot(self.wave, self.ill)
            plt.show()
        else:
            print('No standard illuminant loaded')