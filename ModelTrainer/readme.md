koi_kepmag: Kepler-band brightness (magnitude) - unit: mag (magnitude).
pl_radj: Planet radius (Jupiter radii) - unit: R_J (Jupiter radius, after converting from Earth radii if needed).
koi_impact: Impact parameter - unit: dimensionless.
pl_trandur: Transit duration - unit: hours.
depth: Transit depth - unit: fraction (after normalizing from ppm or percent).
pl_orbper: Orbital period - unit: days.
st_teff: Stellar effective temperature - unit: K (Kelvin).
st_logg: Stellar surface gravity (log scale) - unit: dex (log10(cm/s²)).
st_rad: Stellar radius (stellar radius) - unit: R_Sun (radius of the Sun).
pl_insol: Insolation flux (insolation flux) - unit: F_Earth (ratio to Earth flux, unitless relative).
pl_eqt: Equilibrium temperature (planetary equilibrium temperature) - unit: K (Kelvin).
st_dist: Stellar distance (stellar distance) - unit: pc (parsec).
density_proxy: Density proxy (derived: 1 / pl_radj³) - unit: unitless (proxy, based on Jupiter radius).
habitability_proxy: Habitability proxy (derived: pl_orbper * 0.7 / st_teff) - units: days/K (days per Kelvin, but used as a unitless proxy).
transit_shape_proxy: Transit shape proxy (derived: depth / pl_trandur) - units: fraction/hour (relative unitless).