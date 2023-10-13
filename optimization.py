
from ase import Atoms
from ase.build import fcc100, add_vacuum
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.data import atomic_numbers
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.io import read, write
from ase.optimize.sciopt import SciPyFminCG
from ase.visualize import view

import numpy as np
import os

surface_size = (6, 6, 3)
slab = fcc100('Au', surface_size, periodic=True)
add_vacuum(slab, 10)
slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

# define the volume in which the adsorbed cluster is optimized
# the volume is defined by a corner position (p0)
# and three spanning vectors (v1, v2, v3)
pos = slab.get_positions()
cell = slab.get_cell()
p0 = np.array([cell[0, 0]/4, cell[1, 1]/4, max(pos[:, 2]) + 2.])
v1 = cell[0, :] * 0.5
v2 = cell[1, :] * 0.5
v3 = cell[2, :] * .3

# Define the composition of the atoms to optimize
atom_numbers = 12 * [atomic_numbers['Pt']]

# define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                    ratio_of_covalent_radii=0.7)

# create the starting population
sg = StartGenerator(slab, atom_numbers, blmin,
                    box_to_place_in=[p0, [v1, v2, v3]])
# view(slab)


atoms = sg.get_new_candidate()
atoms.set_calculator(EMT())
atoms.pbc = [True] * 3
traj_name = 'images/pt12_au100_emt.traj'
if not os.path.isfile(traj_name):
    dyn = SciPyFminCG(atoms, logfile='-', trajectory=traj_name)
    dyn.run(fmax=0.05)

traj = read(traj_name, index=':')
view(traj)  #, viewer='ngl')

# write(traj_name.replace('.traj', '.gif'), traj[30:-1:2], interval=100, rotation='-50x')