import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser, ShrakeRupley
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem
import mdtraj as md
import platform
import os
import pkg_resources
from typing import Generator
from rdkit.Geometry import Point3D


# Step 1: Load the topology
temp_topology = "gro file"
topology = md.load(temp_topology).topology

# Step 2: Select protein and ligand atoms (exclude solvent/ions)
atom_indices = [
    atom.index for atom in topology.atoms
    if atom.residue.name not in ["HOH", "WAT", "T3P"]
]

# Step 3: Load and restrict the trajectory
traj_dir = r"trajectory file"
try:
    traj = md.load_dtr(traj_dir, top=temp_topology)
    traj = traj.atom_slice(atom_indices)
except Exception as e:
    print(f"Failed to load trajectory: {str(e)}")
    raise

# Step 4: Save the stripped system
full_system_gro = "stripped gro file"
traj.save_gro(full_system_gro)

# Verify atom count
print(f"Topology atoms: {topology.n_atoms}")
print(f"Trajectory atoms after slicing: {traj.n_atoms}")

class BindingAffinityPredictor:
    def __init__(self, topology_file: str, trajectory_dir: str):
        try:
            self.topology_file = topology_file
            self.trajectory_dir = trajectory_dir
            self.topology = md.load(topology_file).topology
            self.trajectory = md.load_dtr(trajectory_dir, top=topology_file)
            atom_indices = [
                atom.index for atom in self.topology.atoms
                if atom.residue.name not in ["HOH", "WAT", "T3P"]
            ]
            self.trajectory = self.trajectory.atom_slice(atom_indices)
            print(f"BindingAffinityPredictor - Topology atoms: {self.topology.n_atoms}")
            print(f"BindingAffinityPredictor - Trajectory atoms after slicing: {self.trajectory.n_atoms}")
        except Exception as e:
            raise ValueError(f"Error initializing predictor: {str(e)}")

    def _trajectory_frames(self) -> Generator:
        try:
            for frame in range(len(self.trajectory)):
                yield self.trajectory[frame]
        except Exception as e:
            raise RuntimeError(f"Error yielding trajectory frame: {str(e)}")

    def process_trajectory(self):
        try:
            for frame in self._trajectory_frames():
                temp_pdb = "temp_frame.pdb"
                frame.save_pdb(temp_pdb)
                print(f"Processed frame with {frame.n_atoms} atoms")
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
        except Exception as e:
            raise RuntimeError(f"Error processing trajectory: {str(e)}")

class ProteinLigandFeaturizer:
    def __init__(self, radius=10.0, resolution=1.0):
        self.radius = radius
        self.resolution = resolution
        self.hydrophobic = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "PRO", "TRP"}

    def parse_pdb(self, topology_file):
        try:
            if topology_file.endswith(".gro"):
                temp_pdb = "temp_topology.pdb"
                traj = md.load(topology_file)
                traj.save_pdb(temp_pdb)
                topology_file = temp_pdb

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", topology_file)
            protein = []
            ligand = []
            for residue in structure.get_residues():
                if residue.get_resname() == "UNK" and len(list(residue.get_atoms())) >= 40: # Replace with actual ligand name
                    ligand.append(residue)
                elif residue.id[0].strip() == "":
                    protein.append(residue)
            
            if not ligand:
                raise ValueError("No ligand found in topology file")
            return protein, ligand[0]
        except Exception as e:
            raise ValueError(f"Error parsing topology file: {str(e)}")
        finally:
            if 'temp_pdb' in locals() and os.path.exists(temp_pdb):
                os.remove(temp_pdb)

    def _residues_to_mol(self, residues):
        pdb_block = ""
        atom_serial = 1
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'
        }
        atom_name_to_element = {
            'CL': 'Cl', 'BR': 'Br', 'ZN': 'Zn', 'FE': 'Fe', 'MG': 'Mg', 'CA': 'Ca', 'NA': 'Na',
            'LP': 'C', 'X': 'C'
        }
        for residue in residues:
            res_id = residue.get_id()[1]
            res_name = residue.get_resname()
            for atom in residue.get_atoms():
                x, y, z = atom.get_coord()
                atom_name = atom.get_name().strip()
                if len(atom_name) > 4:
                    atom_name = atom_name[:4]
                elif len(atom_name) < 4:
                    atom_name = f" {atom_name:<3}"
                element = atom.element if atom.element else atom_name_to_element.get(atom_name.strip().upper(), atom_name.strip()[0].upper())
                if element not in valid_elements:
                    print(f"Invalid element '{element}' for atom {atom_name} in residue {res_name} (serial {atom_serial}), defaulting to 'C'")
                    element = 'C'
                if atom_serial == 9157:
                    print(f"Atom 9157: name={atom_name}, element={element}, residue={res_name}, res_id={res_id}")
                pdb_block += f"ATOM  {atom_serial:5d} {atom_name:<4} {res_name:<3} A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
                atom_serial += 1
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

# Store coordinates as props so `.GetDoubleProp("x")` works
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom.SetDoubleProp("x", pos.x)
            atom.SetDoubleProp("y", pos.y)
            atom.SetDoubleProp("z", pos.z)
        if mol is None:
            raise ValueError("Failed to convert residues to RDKit molecule")
        try:
            AllChem.SanitizeMol(mol)
            AllChem.AssignAtomChiralTagsFromStructure(mol)
        except Exception as e:
            print(f"Sanitization failed for protein molecule: {str(e)}. Proceeding without sanitization.")
        return mol

    def _residue_to_mol(self, residue):
        pdb_block = ""
        atom_serial = 1
        res_id = residue.get_id()[1]
        res_name = residue.get_resname()
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'
        }
        atom_name_to_element = {
            'CL': 'Cl', 'BR': 'Br', 'ZN': 'Zn', 'FE': 'Fe', 'MG': 'Mg', 'CA': 'Ca', 'NA': 'Na',
            'LP': 'C', 'X': 'C'
        }
        for atom in residue.get_atoms():
            x, y, z = atom.get_coord()
            atom_name = atom.get_name().strip()
            if len(atom_name) > 4:
                atom_name = atom_name[:4]
            elif len(atom_name) < 4:
                atom_name = f" {atom_name:<3}"
            element = atom.element if atom.element else atom_name_to_element.get(atom_name.strip().upper(), atom_name.strip()[0].upper())
            if element not in valid_elements:
                print(f"Invalid element '{element}' for atom {atom_name} in residue {res_name}, defaulting to 'C'")
                element = 'C'
            pdb_block += f"HETATM{atom_serial:5d} {atom_name:<4} {res_name:<3} A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
            atom_serial += 1
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, atom in enumerate(mol.GetAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

# Store coordinates as props so `.GetDoubleProp("x")` works
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom.SetDoubleProp("x", pos.x)
            atom.SetDoubleProp("y", pos.y)
            atom.SetDoubleProp("z", pos.z)
        if mol is None:
            raise ValueError("Failed to convert ligand to RDKit molecule")
        try:
            AllChem.SanitizeMol(mol)
            AllChem.AssignAtomChiralTagsFromStructure(mol)
        except Exception as e:
            print(f"Sanitization failed for ligand molecule: {str(e)}. Proceeding without sanitization.")
        return mol
    def _distance(self, pos1, pos2):
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

    def _lennard_jones(self, dist, epsilon=0.1, sigma=3.0):
        r = dist / sigma
        return 4 * epsilon * ((1/r)**12 - (1/r)**6)

    def _is_hydrophobic(self, atom):
        residue = atom.get_parent()
        return 1.0 if residue.get_resname() in self.hydrophobic else 0.0

    def _is_donor(self, atom):
        return 1.0 if atom.element in ["N", "O"] and any(n.element == "H" for n in atom.get_parent().get_atoms()) else 0.0

    def _is_acceptor(self, atom):
        return 1.0 if atom.element in ["O", "N"] else 0.0
    

    def calculate_interaction_features(self, protein, ligand):
        try:
            prot_mol = self._residues_to_mol(protein)
            lig_mol = self._residue_to_mol(ligand)
            fp = plf.Fingerprint()
            prolif_version = pkg_resources.get_distribution("prolif").version
            print(f"[DEBUG] ProLIF version: {prolif_version}")

            prot_plf = plf.Molecule.from_rdkit(prot_mol)
            lig_plf = plf.Molecule.from_rdkit(lig_mol)
             
            fp.run_from_iterable([lig_plf], prot_plf)
            df = fp.to_dataframe()
            if df.empty:
               print("[⚠️] No interactions detected — filling fingerprint with zeros.")
               fp_vector = np.zeros(128, dtype=np.float32)  # or your desired fixed length
            else:
               fp_vector = df.values.flatten().astype(np.float32)

            elec, vdw = 0.0, 0.0
            for patom in prot_mol.GetAtoms():
                pos1 = np.array([patom.GetDoubleProp("x"), patom.GetDoubleProp("y"), patom.GetDoubleProp("z")])
                q1 = patom.GetDoubleProp("PartialCharge") if patom.HasProp("PartialCharge") else 0.0
                for latom in lig_mol.GetAtoms():
                    pos2 = np.array([latom.GetDoubleProp("x"), latom.GetDoubleProp("y"), latom.GetDoubleProp("z")])
                    q2 = latom.GetDoubleProp("PartialCharge") if latom.HasProp("PartialCharge") else 0.0
                    dist = self._distance(pos1, pos2)
                    elec += (q1 * q2) / (dist + 1e-6)
                    vdw += self._lennard_jones(dist)
            physics = np.array([elec, vdw, 0.0], dtype=np.float32)
            return fp_vector, physics
        except Exception as e:
            raise ValueError(f"Error calculating interaction features: {str(e)}")

    def voxelize(self, protein, ligand):
        try:
            lig_coords = [atom.get_coord() for atom in ligand.get_atoms()]
            center = np.mean(lig_coords, axis=0)
            grid_size = int(2 * self.radius / self.resolution) + 1
            origin = center - self.radius
            channels = 10
            grid = np.zeros((channels, grid_size, grid_size, grid_size), dtype=np.float32)
            sr = ShrakeRupley()
            sr.compute(ligand.get_parent().get_parent(), level="A")
            for residue in protein + [ligand]:
                for atom in residue.get_atoms():
                    coord = atom.get_coord()
                    idx = ((coord - origin) / self.resolution).astype(int)
                    if np.all((0 <= idx) & (idx < grid_size)):
                        self._update_voxel(grid, atom, idx)
            return grid
        except Exception as e:
            raise ValueError(f"Error voxelizing structure: {str(e)}")

    def _update_voxel(self, grid, atom, idx):
        el = atom.element
        channel_map = {"C": 0, "N": 1, "O": 2, "S": 3}
        if el in channel_map:
            grid[channel_map[el], idx[0], idx[1], idx[2]] += 1.0
        else:
            grid[4, idx[0], idx[1], idx[2]] += 1.0
        grid[4, idx[0], idx[1], idx[2]] += atom.get_occupancy()
        grid[5, idx[0], idx[1], idx[2]] += self._is_hydrophobic(atom)
        grid[6, idx[0], idx[1], idx[2]] += self._is_donor(atom)
        grid[7, idx[0], idx[1], idx[2]] += self._is_acceptor(atom)
        grid[8, idx[0], idx[1], idx[2]] += atom.sasa if hasattr(atom, "sasa") else 0.0
        grid[9, idx[0], idx[1], idx[2]] += atom.get_occupancy()

class HybridBindingModel(nn.Module):
    def __init__(self, voxel_channels=10, fp_dim=128, physics_dim=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(voxel_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.fp_net = nn.Sequential(
            nn.Linear(fp_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        self.physics_net = nn.Sequential(
            nn.Linear(physics_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regressor = nn.Sequential(
            nn.Linear(64*5*5*5 + 32 + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, voxel, fingerprint, physics):
        try:
            cnn_feat = self.cnn(voxel).flatten(1)
            fp_feat = self.fp_net(fingerprint)
            physics_feat = self.physics_net(physics)
            combined = torch.cat([cnn_feat, fp_feat, physics_feat], dim=1)
            return self.regressor(combined)
        except Exception as e:
            raise RuntimeError(f"Error in model forward pass: {str(e)}")

class MDTrajectoryProcessor:
    def __init__(self, topology, trajectory_dir, frame_limit=None):
        try:
            self.frame_limit = frame_limit
            self.topology = md.load(topology).topology
            self.trajectory = md.load_dtr(trajectory_dir, top=topology)
            atom_indices = [
                atom.index for atom in self.topology.atoms
                if atom.residue.name not in ["HOH", "WAT", "T3P"]
            ]
            self.trajectory = self.trajectory.atom_slice(atom_indices)
            self.featurizer = ProteinLigandFeaturizer()
            print(f"MDTrajectoryProcessor - Topology atoms: {self.topology.n_atoms}")
            print(f"MDTrajectoryProcessor - Trajectory atoms after slicing: {self.trajectory.n_atoms}")
        except Exception as e:
            raise ValueError(f"Error loading MD trajectory: {str(e)}")
        
    def process_frame(self, frame_idx):
        try:
            temp_pdb = "temp_frame.pdb"
            self.trajectory[frame_idx].save_pdb(temp_pdb)
            print(f"Lines from temp_frame.pdb (frame {frame_idx}):")
            atom_count = 0
            with open(temp_pdb, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_count += 1
                        if 9150 <= atom_count <= 9160 or "UNK" in line:
                            print(line.strip())
            print(f"Total atoms in temp_frame.pdb: {atom_count}")
            protein, ligand = self.featurizer.parse_pdb(temp_pdb)
            fp, physics = self.featurizer.calculate_interaction_features(protein, ligand)
            voxel = self.featurizer.voxelize(protein, ligand)
            energies = {
                'vdw': physics[1],
                'electrostatic': physics[0],
                'solvation': physics[2]
                
            }
           
            delta_g = physics[0] + physics[1]
            

            return {
                'voxel': torch.tensor(voxel, dtype=torch.float32),
                'fingerprint': torch.tensor(fp, dtype=torch.float32),
                'physics': torch.tensor(list(energies.values()), dtype=torch.float32),
                'frame_energy': torch.tensor([delta_g], dtype=torch.float32)
            }
        except Exception as e:
            raise ValueError(f"Error processing frame {frame_idx}: {str(e)}")
        finally:
            if os.path.exists(temp_pdb):
                os.remove(temp_pdb)

    def _calculate_vdw_energy(self, protein, ligand):
        return 0.0

    def _calculate_electrostatic(self, protein, ligand):
        return 0.0

    def _estimate_solvation(self, protein, ligand):
        return 0.0

class BindingDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_files, md_trajectory_dirs):
        self.samples = []
        for pdb, traj_dir in zip(pdb_files, md_trajectory_dirs):
            processor = MDTrajectoryProcessor(pdb, traj_dir)
            for frame in range(min(10, len(processor.trajectory))):
                self.samples.append(processor.process_frame(frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (sample['voxel'], sample['fingerprint'], sample['physics']), sample['frame_energy']

def estimate_entropy(trajectory):
    try:
        coords = trajectory.xyz
        cov = np.cov(coords.reshape(coords.shape[0], -1).T)
        eigenvalues = np.linalg.eigvalsh(cov)
        entropy = 0.5 * np.sum(np.log(eigenvalues + 1e-10)) * 0.001987 * 300
        return entropy
    except Exception as e:
        print(f"Error estimating entropy: {str(e)}")
        return 0.0

def calculate_binding_energy(model, trajectory_processor):
    try:
        energies = []
        limit = trajectory_processor.frame_limit or len(trajectory_processor.trajectory)
        for frame in range(limit):
            sample = trajectory_processor.process_frame(frame)
            with torch.no_grad():
                pred = model(sample['voxel'].unsqueeze(0),
                             sample['fingerprint'].unsqueeze(0),
                             sample['physics'].unsqueeze(0))
                energies.append(pred.item())
        
        
        delta_g = np.mean(energies)
        std_dev = np.std(energies)

        entropy = estimate_entropy(trajectory_processor.trajectory)
        return delta_g - entropy, std_dev
    except Exception as e:
        raise RuntimeError(f"Error calculating binding energy: {str(e)}")
    
def _estimate_solvation(self, protein, ligand):
    sr = ShrakeRupley()
    sr.compute(protein.get_parent().get_parent(), level="A")
    sasa_complex = sum(atom.sasa for atom in protein.get_atoms() if hasattr(atom, "sasa"))
    sasa_ligand = sum(atom.sasa for atom in ligand.get_atoms() if hasattr(atom, "sasa"))
    return 0.025 * (sasa_complex + sasa_ligand)


def train_model(pdb_file, traj_dir):
    try:
        model = HybridBindingModel(fp_dim=128)
        dataset = BindingDataset([pdb_file], [traj_dir])
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(10):
            total_loss = 0.0
            for (voxels, fps, physics), energies in loader:
                optimizer.zero_grad()
                outputs = model(voxels, fps, physics)
                loss = criterion(outputs, energies)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), "hybrid_model.pth")
        return model
    except Exception as e:
        raise RuntimeError(f"Error training model: {str(e)}")

def main():
    try:
        topo_file = "gro file"
        traj_dir = r"trajectory file"
        model = train_model(topo_file, traj_dir)
        processor = MDTrajectoryProcessor(topo_file, traj_dir, frame_limit=10)
        delta_g, std_dev = calculate_binding_energy(model, processor)
        print(f"Predicted Binding Free Energy: {delta_g:.2f} kcal/mol, Std Dev: {std_dev:.2f}")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        if os.path.exists(full_system_gro):
            os.remove(full_system_gro)

if __name__ == "__main__":
    if platform.system() != "Emscripten":
        main()
    else:
        import asyncio
        async def async_main():
            main()
        asyncio.ensure_future(async_main())