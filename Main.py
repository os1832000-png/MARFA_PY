"""
MARFA - Complete Python Implementation with Multi-Molecule Support

This version calculates and plots absorption for all your molecules
(SO2, NO2, NO, N2O, CO, NH3, H2O, O2, O3, CH4, HNO3, CO2) in a single graph.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import os
from scipy.special import wofz
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

class Constants:
    """Physical constants matching Fortran MARFA"""
    c = 2.99792458e10          # speed of light [cm/s]
    h = 6.62606957e-27         # Planck constant [erg·s]
    k = 1.380649e-23           # Boltzmann constant [J/K]
    Na = 6.02214129e23         # Avogadro constant [1/mol]
    c2 = 1.4388                # second radiation constant [cm·K]
    T_ref = 296.0              # reference temperature [K]
    P_ref = 1.0                # reference pressure [atm]
    pi = np.pi
    sqrt_pi = np.sqrt(np.pi)
    sqrt_ln2 = np.sqrt(np.log(2.0))
    atm_to_pa = 101325.0       # 1 atm = 101325 Pa


# ==============================================================================
# MOLECULE PROPERTIES
# ==============================================================================

MOLECULE_NAMES = {
    1: "H2O", 2: "CO2", 3: "O3", 4: "N2O", 5: "CO",
    6: "CH4", 7: "O2", 8: "NO", 9: "SO2", 10: "NO2",
    11: "NH3", 12: "HNO3"
}

DEFAULT_MASSES = {
    1: 18.015, 2: 44.010, 3: 47.998, 4: 44.013,
    5: 28.010, 6: 16.043, 7: 31.999, 8: 30.006,
    9: 64.066, 10: 46.006, 11: 17.031, 12: 63.012
}

# Colors for plotting each molecule
MOLECULE_COLORS = {
    1: '#1f77b4',   # H2O - blue
    2: '#ff7f0e',   # CO2 - orange
    3: '#2ca02c',   # O3 - green
    4: '#d62728',   # N2O - red
    5: '#9467bd',   # CO - purple
    6: '#8c564b',   # CH4 - brown
    7: '#e377c2',   # O2 - pink
    8: '#7f7f7f',   # NO - gray
    9: '#bcbd22',   # SO2 - yellow-green
    10: '#17becf',  # NO2 - cyan
    11: '#ff9896',  # NH3 - light red
    12: '#9edae5'   # HNO3 - light cyan
}


# ==============================================================================
# SPECTRAL LINE CLASS
# ==============================================================================

@dataclass
class SpectralLine:
    """HITRAN spectral line parameters"""
    nu: float          # line center wavenumber [cm⁻¹]
    S: float           # line intensity [cm⁻¹/(molecule·cm⁻²)]
    gamma_air: float   # air-broadened HWHM [cm⁻¹/atm]
    gamma_self: float  # self-broadened HWHM [cm⁻¹/atm]
    E_low: float       # lower state energy [cm⁻¹]
    n_air: float       # temperature exponent
    delta_air: float   # pressure shift [cm⁻¹/atm]
    mol_id: int        # molecule ID
    iso_id: int        # isotopologue ID


# ==============================================================================
# TIPS DATABASE
# ==============================================================================

class TIPS:
    """Total Internal Partition Sums"""
    
    def __init__(self):
        self.temps = np.linspace(1, 3000, 3000)
        q296 = {
            1: 178.12, 2: 289.49, 3: 4870.3, 4: 1122.3,
            5: 108.58, 6: 590.43, 7: 216.21, 8: 159.47,
            9: 5792.6, 10: 2379.5, 11: 169.24, 12: 11456.0
        }
        
        self.q_data = {}
        for mol_id in range(1, 13):
            q_ref = q296.get(mol_id, 100.0)
            self.q_data[mol_id] = q_ref * (self.temps / 296.0) ** 1.5
    
    def get_q(self, mol_id: int, temperature: float) -> float:
        """Get partition sum for molecule at temperature"""
        if mol_id not in self.q_data:
            mol_id = 2
        temp = np.clip(temperature, 1.0, 3000.0)
        idx = int(temp) - 1
        idx = max(0, min(idx, len(self.temps) - 2))
        t1, t2 = self.temps[idx], self.temps[idx + 1]
        q1, q2 = self.q_data[mol_id][idx], self.q_data[mol_id][idx + 1]
        return q1 + (q2 - q1) * (temp - t1) / (t2 - t1)


# ==============================================================================
# LINE SHAPES
# ==============================================================================

class LineShapes:
    """Spectral line shape functions"""
    
    @staticmethod
    def voigt(x: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
        """Voigt profile using Faddeeva function"""
        if gamma == 0:
            return LineShapes.doppler(x, alpha)
        if alpha == 0:
            return LineShapes.lorentz(x, gamma)
        
        sigma = alpha / np.sqrt(np.log(2.0))
        z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
        w = wofz(z)
        return np.real(w) / (sigma * np.sqrt(2.0 * np.pi))
    
    @staticmethod
    def lorentz(x: np.ndarray, gamma: float) -> np.ndarray:
        """Lorentz profile"""
        return gamma / (np.pi * (x**2 + gamma**2))
    
    @staticmethod
    def doppler(x: np.ndarray, alpha: float) -> np.ndarray:
        """Doppler profile"""
        sqrt_ln2_pi = np.sqrt(np.log(2.0) / np.pi)
        return (sqrt_ln2_pi / alpha) * np.exp(-np.log(2.0) * (x / alpha)**2)


# ==============================================================================
# HITRAN FILE READER
# ==============================================================================

class HITRANReader:
    """Read HITRAN .par format files"""
    
    @staticmethod
    def read_par_file(filename: str, 
                     molecule_id: Optional[int] = None,
                     wavenumber_min: Optional[float] = None,
                     wavenumber_max: Optional[float] = None,
                     intensity_threshold: Optional[float] = None) -> List[SpectralLine]:
        """Read HITRAN .par format file"""
        lines = []
        
        if not os.path.exists(filename):
            print(f"  ⚠ File {filename} not found")
            return lines
        
        try:
            with open(filename, 'r') as f:
                for line_str in f:
                    if len(line_str) < 160:
                        continue
                    
                    try:
                        mol = int(line_str[0:2])
                        iso = int(line_str[2])
                        nu = float(line_str[3:15])
                        S = float(line_str[15:25])
                        gamma_air = float(line_str[35:40])
                        gamma_self = float(line_str[40:45])
                        E_low = float(line_str[45:55])
                        n_air = float(line_str[55:59])
                        delta_air = float(line_str[59:67])
                        
                        if molecule_id is not None and mol != molecule_id:
                            continue
                        if wavenumber_min is not None and nu < wavenumber_min:
                            continue
                        if wavenumber_max is not None and nu > wavenumber_max:
                            continue
                        if intensity_threshold is not None and S < intensity_threshold:
                            continue
                        
                        line = SpectralLine(
                            nu=nu, S=S, gamma_air=gamma_air, gamma_self=gamma_self,
                            E_low=E_low, n_air=n_air, delta_air=delta_air,
                            mol_id=mol, iso_id=iso
                        )
                        lines.append(line)
                        
                    except (ValueError, IndexError):
                        continue
            
            if lines:
                print(f"  ✓ {MOLECULE_NAMES.get(molecule_id, 'Unknown')}: {len(lines)} lines")
            
        except Exception as e:
            print(f"  ✗ Error reading {filename}: {e}")
        
        return lines
    
    @staticmethod
    def read_all_molecules(data_dir: str = ".",
                          wavenumber_range: Optional[Tuple[float, float]] = None,
                          intensity_threshold: float = 1e-30) -> Dict[int, List[SpectralLine]]:
        """Read all HITRAN files in directory"""
        print(f"\n📁 Reading HITRAN files from: {data_dir}")
        print("=" * 70)
        
        molecule_files = {
            1: "H2O.par",   2: "CO2.par",   3: "O3.par",    4: "N2O.par",
            5: "CO.par",    6: "CH4.par",   7: "O2.par",    8: "NO.par",
            9: "SO2.par",  10: "NO2.par",  11: "NH3.par",  12: "HNO3.par"
        }
        
        all_lines = {}
        
        for mol_id, filename in molecule_files.items():
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                nu_min, nu_max = wavenumber_range if wavenumber_range else (None, None)
                lines = HITRANReader.read_par_file(
                    filepath, molecule_id=mol_id,
                    wavenumber_min=nu_min, wavenumber_max=nu_max,
                    intensity_threshold=intensity_threshold
                )
                if lines:
                    all_lines[mol_id] = lines
        
        total_lines = sum(len(lines) for lines in all_lines.values())
        print(f"\n📊 Total lines loaded: {total_lines:,}")
        print("=" * 70)
        
        return all_lines
    
# ==============================================================================
# MARFA CALCULATOR
# ==============================================================================

class MARFA:
    """Main MARFA spectroscopy calculator"""
    
    def __init__(self):
        self.const = Constants()
        self.tips = TIPS()
        self.shapes = LineShapes()
        
    def calculate_doppler_width(self, nu0: float, T: float, mol_id: int) -> float:
        """Calculate Doppler HWHM"""
        M = DEFAULT_MASSES.get(mol_id, 44.0)
        m = M / self.const.Na
        return nu0 * np.sqrt(2.0 * self.const.k * T * np.log(2.0) / (m * self.const.c**2))
    
    def calculate_lorentz_width(self, line: SpectralLine, P: float, P_self: float, T: float) -> float:
        """Calculate Lorentz HWHM"""
        pressure_broadening = (line.gamma_air * (P - P_self) + line.gamma_self * P_self)
        temperature_factor = (self.const.T_ref / T) ** line.n_air
        return pressure_broadening * temperature_factor
    
    def calculate_line_intensity_at_T(self, line: SpectralLine, T: float, P: float) -> float:
        """Calculate temperature-dependent line intensity"""
        nu_shifted = line.nu + line.delta_air * P
        Q_T = self.tips.get_q(line.mol_id, T)
        Q_ref = self.tips.get_q(line.mol_id, self.const.T_ref)
        
        partition_ratio = Q_ref / Q_T if Q_T > 0 else 1.0
        
        c2_E_T = self.const.c2 * line.E_low / T
        c2_E_ref = self.const.c2 * line.E_low / self.const.T_ref
        boltzmann_ratio = np.exp(-c2_E_T) / np.exp(-c2_E_ref)
        
        c2_nu_T = self.const.c2 * nu_shifted / T
        c2_nu_ref = self.const.c2 * nu_shifted / self.const.T_ref
        
        if c2_nu_T > 50:
            emission_ratio = 1.0
        else:
            emission_ratio = (1.0 - np.exp(-c2_nu_T)) / (1.0 - np.exp(-c2_nu_ref))
        
        return line.S * partition_ratio * boltzmann_ratio * emission_ratio
    
    def calculate_absorption(self,                                      #Important
                           nu_grid: np.ndarray,
                           lines: List[SpectralLine],
                           T: float,
                           P: float,
                           mole_fraction: float = 1.0,                   
                           line_cutoff: float = 25.0) -> np.ndarray:
        """Calculate absorption coefficient"""
        if not lines:
            return np.zeros_like(nu_grid)
        
        kappa = np.zeros_like(nu_grid)
        P_self = P * mole_fraction
        # n_total = P * self.const.atm_to_pa / (self.const.k * T) # Old method
        k_SI = 1.380649e-23  # J/K
        n_total = (P * 101325.0) / (k_SI * T)  # molecules/m^3
        n_total = n_total / 1e6                # molecules/cm^3

        mol_id = lines[0].mol_id
        
        for line in lines:
            S_T = self.calculate_line_intensity_at_T(line, T, P)
            if S_T <= 0:
                continue
            
            alpha = self.calculate_doppler_width(line.nu, T, mol_id)
            gamma = self.calculate_lorentz_width(line, P, P_self, T)
            width = max(gamma, alpha)
            
            if width <= 0:
                continue
            
            cutoff = line_cutoff * width
            mask = np.abs(nu_grid - line.nu) <= cutoff
            
            if not np.any(mask):
                continue
            
            x = nu_grid[mask] - line.nu
            shape = self.shapes.voigt(x, gamma, alpha)
            kappa[mask] += S_T * shape * n_total * mole_fraction
        
        return kappa


# ==============================================================================
# ATMOSPHERIC COMPOSITION
# ==============================================================================

def get_standard_composition():
    """Standard atmospheric composition"""
    return {
        2: 400e-6,   # CO2: 400 ppm 
        7: 0.2095,   # O2: 20.95%
        4: 0.32e-6,  # N2O: 0.32 ppm
        5: 0.2e-6,   # CO: 0.2 ppm
        6: 1.8e-6,   # CH4: 1.8 ppm
        8: 0.1e-6,   # NO: 0.1 ppm
        9: 1e-6,     # SO2: 1 ppm (elevated for visibility)
        10: 0.5e-6,  # NO2: 0.5 ppm
        3: 0.5e-6,   # O3: 0.5 ppm
        1: 0.01,     # H2O: 1%
        11: 0.5e-6,  # NH3: 0.5 ppm
        12: 0.5e-6   # HNO3: 0.5 ppm
    }


# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================

def plot_multi_molecule_absorption(nu_grid: np.ndarray, 
                                   individual_absorption: Dict[int, np.ndarray],
                                   total_absorption: np.ndarray,
                                   T: float, P: float,
                                   save_path: str = "marfa_all_molecules.png"):
    """Plot all molecules in a single graph matching reference style"""
    
    print("\n📈 Generating plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot total absorption with thin blue lines (matching reference image)
    ax.plot(nu_grid, total_absorption, 'b-', linewidth=0.8)
    
    # Set labels (matching reference style)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel('Absorption Coefficient', fontsize=11)
    
    # Set x-axis limits to match reference
    ax.set_xlim(nu_grid[0], nu_grid[-1])
    
    # Set y-axis to start from 0 (matching reference)
    ax.set_ylim(bottom=0, top=total_absorption.max() * 1.05)
    
    # Format y-axis in scientific notation (matching reference 1e-21)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    
    # Add grid (matching reference)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {save_path}")
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_co2_only_absorption():
    print("\n" + "=" * 70)
    print("  MARFA - CO2 Absorption Coefficient Only")
    print("=" * 70)

    # ----------------------------
    # Configuration (EDIT HERE)
    # ----------------------------
    DATA_DIR = "."            # folder containing CO2.par
    CO2_FILE = os.path.join(DATA_DIR, "CO2.par")

    NU_MIN = 1000.0
    NU_MAX = 4000.0
    NU_RESOLUTION = 0.01

    T = 296.0   # K
    P = 0.5     # atm

    # CO2 mole fraction (400 ppm default)
    X_CO2 = 400e-6

    print(f"\n⚙️  Configuration:")
    print(f"   Molecule: CO2")
    print(f"   Wavenumber range: {NU_MIN} - {NU_MAX} cm⁻¹")
    print(f"   Resolution: {NU_RESOLUTION} cm⁻¹")
    print(f"   Temperature: {T} K")
    print(f"   Pressure: {P} atm")
    print(f"   CO2 mole fraction: {X_CO2:.6e} ({X_CO2*1e6:.1f} ppm)")

    # Wavenumber grid
    nu_grid = np.arange(NU_MIN, NU_MAX + NU_RESOLUTION, NU_RESOLUTION)
    print(f"   Grid points: {len(nu_grid):,}")

    # Read ONLY CO2 lines in range
    print("\n📖 Reading CO2 HITRAN lines...")
    co2_lines = HITRANReader.read_par_file(
        CO2_FILE,
        molecule_id=2,
        wavenumber_min=NU_MIN,
        wavenumber_max=NU_MAX,
        intensity_threshold=1e-27
    )

    if not co2_lines:
        print("\n⚠️  No CO2 lines loaded. Check CO2.par exists and range is correct.")
        return

    # Compute absorption coefficient
    marfa = MARFA()
    print("\n🔬 Computing CO2 absorption coefficient...")
    kappa_co2 = marfa.calculate_absorption(
        nu_grid=nu_grid,
        lines=co2_lines,
        T=T,
        P=P,
        mole_fraction=X_CO2,
        line_cutoff=25.0
    )

    print(f"✓ CO2 lines used: {len(co2_lines):,}")
    print(f"✓ Max absorption: {kappa_co2.max():.3e} cm⁻¹")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(nu_grid, kappa_co2, linewidth=0.8)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorption Coefficient (cm⁻¹)")
    plt.title(f"CO2 Absorption Coefficient ({NU_MIN}-{NU_MAX} cm⁻¹)")
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig("marfa_CO2_3550_3750.png", dpi=300)
    plt.show()

    # Save data
    np.savetxt(
        "marfa_CO2_3550_3750.txt",
        np.column_stack([nu_grid, kappa_co2]),
        header=(
            f"# CO2 absorption coefficient\n"
            f"# T={T} K, P={P} atm, X_CO2={X_CO2} ({X_CO2*1e6:.1f} ppm)\n"
            f"# Range={NU_MIN}-{NU_MAX} cm^-1, dnu={NU_RESOLUTION} cm^-1\n"
            f"# columns: wavenumber(cm^-1)  absorption(cm^-1)"
        ),
        fmt="%.4f %.8e"
    )
    print("💾 Saved: marfa_CO2_3550_3750.txt")
    print("🖼️ Saved: marfa_CO2_3550_3750.png")


if __name__ == "__main__":
    run_co2_only_absorption()

    
