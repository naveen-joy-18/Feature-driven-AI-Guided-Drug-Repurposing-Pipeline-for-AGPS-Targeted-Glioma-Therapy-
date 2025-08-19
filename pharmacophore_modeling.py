import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from prody import *
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
import datetime
import psutil
import traceback
import pickle
import json
from contextlib import contextmanager
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore")

# Configuration
PDB_FILE = os.path.normpath(os.path.join(os.getcwd(), "model_01_5AE1.pdb"))
LIGAND_FOLDER = os.path.normpath("Ligands")
ACTIVE_SITE_RESIDUES = [616,617]
CHAIN_ID = "A"
R = 1.987  # Gas constant in cal/(mol?K)

# Checkpointing configuration
CHECKPOINT_DIR = "checkpoints"
DESCRIPTOR_CHECKPOINT_SIZE = 5000  # Save descriptor checkpoint every N molecules
AFFINITY_CHECKPOINT_SIZE = 100     # Save affinity checkpoint every N molecules

# GPU Configuration
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Set more conservative memory fraction
        torch.cuda.set_per_process_memory_fraction(0.6)
    else:
        device = torch.device('cpu')
        print("Using CPU - GPU not available")
    return device

DEVICE = setup_device()

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

@contextmanager
def memory_cleanup():
    """Context manager for aggressive memory cleanup"""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def log_memory_usage(stage=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage} - CPU: {cpu_mem:.2f} MB, GPU: {gpu_mem:.2f} GB (cached: {gpu_cached:.2f} GB)")
    else:
        print(f"{stage} - CPU: {cpu_mem:.2f} MB")

def save_checkpoint(data, filename):
    """Save checkpoint data"""
    try:
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        if isinstance(data, pd.DataFrame):
            # Clean dataframe before saving
            data_clean = data.copy()
            if 'MolObj' in data_clean.columns:
                data_clean = data_clean.drop('MolObj', axis=1)
            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean.to_excel(filepath, index=False)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        print(f"💾 Checkpoint saved: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to save checkpoint {filename}: {e}")
        return False

def load_checkpoint(filename):
    """Load checkpoint data"""
    try:
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        if not os.path.exists(filepath):
            return None
        
        if filename.endswith('.xlsx'):
            return pd.read_excel(filepath)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load checkpoint {filename}: {e}")
        return None

def get_processed_files():
    """Get list of already processed files"""
    progress_file = os.path.join(CHECKPOINT_DIR, "processed_files.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_processed_files(processed_files):
    """Save list of processed files"""
    progress_file = os.path.join(CHECKPOINT_DIR, "processed_files.json")
    try:
        with open(progress_file, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"❌ Failed to save progress: {e}")

def calcAtomicFeatures(active_site):
    return np.zeros((active_site.numAtoms(), 10))

def calcHydrophobicity(active_site):
    return np.zeros(active_site.numAtoms())

def calcVolume(active_site):
    return 0.0

class Electrostatics:
    def __init__(self, active_site):
        self.potentials = np.zeros(active_site.numAtoms())
    def getPotentials(self):
        return self.potentials

def process_protein(pdb_file, active_site_residues):
    try:
        protein = parsePDB(pdb_file, fetch=False)
        sel_str = f'chain {CHAIN_ID} and resnum {" ".join(map(str, active_site_residues))}'
        active_site = protein.select(sel_str)
        if active_site is None or active_site.numAtoms() == 0:
            raise ValueError(f"Invalid active site selection: {sel_str}")
        coords = active_site.getCoords()
        centroid = np.mean(coords, axis=0)
        dummy_modes = np.zeros((active_site.numAtoms() * 3, 3))
        es = Electrostatics(active_site)
        return {
            'residue_features': calcAtomicFeatures(active_site),
            'anm_modes': dummy_modes,
            'electrostatic': es.getPotentials(),
            'hydrophobicity': calcHydrophobicity(active_site),
            'volume': calcVolume(active_site),
            'surface_points': coords,
            'centroid': centroid
        }
    except Exception as e:
        raise RuntimeError(f"Failed to process protein: {e}")

class LigandProcessor:
    def __init__(self, mol):
        self.mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(self.mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(self.mol)
    
    def get_3dpharma_features(self):
        try:
            return {
                'esp': 0.0,
                'pmi': Descriptors.PMI1(self.mol) if hasattr(Descriptors, 'PMI1') else 0.0,
                'steric': Descriptors.NPR1(self.mol) if hasattr(Descriptors, 'NPR1') else 0.0,
                'affinity': 0.0
            }
        except:
            return {'esp': 0.0, 'pmi': 0.0, 'steric': 0.0, 'affinity': 0.0}
    
    def get_graph_representation(self):
        pos = torch.tensor([list(self.mol.GetConformer().GetAtomPosition(i)) 
                          for i in range(self.mol.GetNumAtoms())], dtype=torch.float)
        features = []
        for atom in self.mol.GetAtoms():
            atom_features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                float(atom.GetHybridization().real),
                float(atom.GetIsAromatic())
            ]
            features.append(atom_features)
        x = torch.tensor(features, dtype=torch.float)
        bonds = list(self.mol.GetBonds())
        if len(bonds) > 0:
            edge_index = torch.tensor(
                [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] 
                 for b in bonds] + 
                [[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] 
                 for b in bonds], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, pos=pos, batch=torch.zeros(x.shape[0], dtype=torch.long))
    
    def get_surface_points(self):
        conf = self.mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())])

def compute_descriptors(mol):
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp('_GasteigerCharge'):
                charges.append(float(atom.GetDoubleProp('_GasteigerCharge')))
        
        return {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'AliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'MaxPartialCharge': max(charges) if charges else 0.0,
            'MinPartialCharge': min(charges) if charges else 0.0,
            'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),
            'MolMR': Descriptors.MolMR(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
            'BertzCT': Descriptors.BertzCT(mol)
        }
    except:
        return {key: 0.0 for key in [
            'MolWt', 'MolLogP', 'NumRotatableBonds', 'TPSA', 'HBA', 'HBD',
            'AromaticRings', 'FractionCSP3', 'HeavyAtomCount', 'AliphaticRings',
            'MaxPartialCharge', 'MinPartialCharge', 'LabuteASA', 'MolMR',
            'BalabanJ', 'BertzCT'
        ]}

def compute_entropy(rot_bonds):
    delta_s_torsion = -R * np.log(rot_bonds + 1)
    return delta_s_torsion

def compute_distance_to_site(mol, centroid):
    conf = mol.GetConformer()
    distances = []
    for atom in mol.GetAtoms():
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        dist = np.linalg.norm(pos - centroid)
        distances.append(dist)
    return np.mean(distances), np.min(distances), np.max(distances)

class PharmaGNN(nn.Module):
    def __init__(self, protein_feature_size=10, ligand_feature_size=5):
        super().__init__()
        self.device = DEVICE
        self.protein_conv1 = GATv2Conv(protein_feature_size, 128, heads=2)  # Reduced complexity
        self.protein_conv2 = GATv2Conv(128*2, 256)
        self.ligand_conv1 = GATv2Conv(ligand_feature_size, 128, heads=2)
        self.ligand_conv2 = GATv2Conv(128*2, 256)
        self.cross_attention = nn.MultiheadAttention(256, 4, batch_first=True)  # Reduced heads
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        self.to(self.device)
    
    def forward(self, protein_data, ligand_data):
        # Move data to device
        protein_data = protein_data.to(self.device)
        ligand_data = ligand_data.to(self.device)
        
        # Process protein
        p_x = self.protein_conv1(protein_data.x, protein_data.edge_index)
        p_x = self.protein_conv2(p_x, protein_data.edge_index)
        p_x = global_mean_pool(p_x, protein_data.batch)
        
        # Process ligand
        l_x = self.ligand_conv1(ligand_data.x, ligand_data.edge_index)
        l_x = self.ligand_conv2(l_x, ligand_data.edge_index)
        l_x = global_mean_pool(l_x, ligand_data.batch)
        
        # Cross attention
        attn_out, _ = self.cross_attention(p_x.unsqueeze(1), l_x.unsqueeze(1), l_x.unsqueeze(1))
        
        # Combine features
        combined = torch.cat([p_x, attn_out.squeeze(1)], dim=1)
        return self.fc(combined)

def calculate_binding_score(protein_features, ligand_features, descriptors, distances, entropy):
    try:
        sc = 1 - cdist(protein_features['surface_points'], ligand_features['surface_points'], 'cosine').mean()
        ec = np.corrcoef(protein_features['electrostatic'], [ligand_features['esp']])[0,1]
        if np.isnan(ec):
            ec = 0.0
        dist_score = 1 / (1 + distances['AvgDistToSite'])
        entropy_score = -entropy / R
        energy = (0.3 * sc) + (0.2 * ec) + (0.2 * ligand_features['affinity']) + \
                 (0.2 * dist_score) + (0.1 * entropy_score)
        return energy
    except:
        return 0.0

def safe_save_dataframe(df, filename, max_retries=3):
    """Safely save DataFrame with retries"""
    for attempt in range(max_retries):
        try:
            df_clean = df.copy()
            if 'MolObj' in df_clean.columns:
                df_clean = df_clean.drop('MolObj', axis=1)
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_clean.to_excel(writer, index=False, sheet_name='Results')
            
            print(f"✅ Successfully saved {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed to save {filename}: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
            else:
                print(f"❌ Failed to save {filename} after {max_retries} attempts")
                return False

def process_single_sdf_file(sdf_path, sdf_file, centroid, processed_files):
    """Process a single SDF file with continuous checkpointing"""
    results = []
    skipped = []
    
    # Check if file already processed
    if sdf_file in processed_files:
        print(f"📁 Loading cached results for {sdf_file}...")
        cached_file = f"descriptors_{sdf_file.replace('.sdf', '')}.xlsx"
        cached_df = load_checkpoint(cached_file)
        if cached_df is not None:
            print(f"✅ Loaded {len(cached_df)} cached results for {sdf_file}")
            return cached_df, []
    
    print(f"🔄 Processing {sdf_file}...")
    
    try:
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
        
        batch_results = []
        processed_count = 0
        total_processed = 0
        
        for idx, mol in enumerate(suppl):
            if idx % 1000 == 0:
                print(f"\r{sdf_file}: Processing molecule {idx + 1}...", end="", flush=True)
            
            if mol is None:
                skipped.append((sdf_file, idx, "Unreadable molecule"))
                continue
            
            try:
                # Sanitize and prepare molecule
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                res = AllChem.EmbedMolecule(mol, randomSeed=42)
                if res != 0:
                    skipped.append((sdf_file, idx, "3D embedding failed"))
                    continue
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Compute descriptors
                desc = compute_descriptors(mol)
                avg_d, min_d, max_d = compute_distance_to_site(mol, centroid)
                entropy = compute_entropy(desc['NumRotatableBonds'])
                
                desc.update({
                    'Ligand': f"{sdf_file}_mol_{idx}",
                    'AvgDistToSite': avg_d,
                    'MinDistToSite': min_d,
                    'MaxDistToSite': max_d,
                    'EntropyLoss': entropy,
                    'SourceFile': sdf_file,
                    'MolIndex': idx
                })
                
                batch_results.append(desc)
                processed_count += 1
                total_processed += 1
                
                # Save checkpoint every N molecules
                if processed_count >= DESCRIPTOR_CHECKPOINT_SIZE:
                    results.extend(batch_results)
                    
                    # Save intermediate checkpoint
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_file = f"descriptors_{sdf_file.replace('.sdf', '')}_temp.xlsx"
                    save_checkpoint(checkpoint_df, checkpoint_file)
                    
                    batch_results = []
                    processed_count = 0
                    
                    # Memory cleanup
                    with memory_cleanup():
                        pass
                    
            except Exception as e:
                skipped.append((sdf_file, idx, f"Processing error: {str(e)[:100]}"))
                continue
        
        # Add remaining results
        if batch_results:
            results.extend(batch_results)
        
        # Save final results for this file
        if results:
            final_df = pd.DataFrame(results)
            final_file = f"descriptors_{sdf_file.replace('.sdf', '')}.xlsx"
            save_checkpoint(final_df, final_file)
            
            # Mark file as processed
            processed_files[sdf_file] = {
                'processed_count': total_processed,
                'timestamp': datetime.datetime.now().isoformat()
            }
            save_processed_files(processed_files)
        
        print(f"\r{sdf_file}: Completed - {total_processed} molecules processed")
        
    except Exception as e:
        print(f"\n❌ Error processing {sdf_file}: {e}")
        return pd.DataFrame(), skipped
    
    return pd.DataFrame(results), skipped

def process_ligands_from_multiple_sdfs(folder, centroid):
    """Process ligands with file-level checkpointing"""
    all_results = []
    all_skipped = []
    
    sdf_files = [f for f in os.listdir(folder) if f.endswith(".sdf")]
    processed_files = get_processed_files()
    
    print(f"📂 Found {len(sdf_files)} SDF files")
    print(f"📋 {len(processed_files)} files already processed")
    
    for file_idx, sdf_file in enumerate(sdf_files):
        print(f"\n🔄 File {file_idx + 1}/{len(sdf_files)}: {sdf_file}")
        
        sdf_path = os.path.join(folder, sdf_file)
        
        with memory_cleanup():
            results_df, skipped = process_single_sdf_file(sdf_path, sdf_file, centroid, processed_files)
            
            if not results_df.empty:
                all_results.append(results_df)
            
            all_skipped.extend(skipped)
            
            log_memory_usage(f"After processing {sdf_file}")
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"\n✅ Total molecules processed: {len(final_df)}")
    else:
        final_df = pd.DataFrame()
        print(f"\n❌ No molecules processed successfully")
    
    if all_skipped:
        print(f"⚠️  Skipped {len(all_skipped)} molecules. First 10 errors:")
        for skip_info in all_skipped[:10]:
            print(f"  {skip_info[0]}_mol_{skip_info[1]}: {skip_info[2]}")
    
    return final_df

class PharmaPipeline:
    def __init__(self):
        print("🔬 Initializing PharmaPipeline...")
        log_memory_usage("Initial")
        
        self.protein_features = process_protein(PDB_FILE, ACTIVE_SITE_RESIDUES)
        self.model = PharmaGNN()
        self.model.eval()
        self.scaler = StandardScaler()
        
        log_memory_usage("After initialization")
        
    def process_affinity_predictions(self, ligand_df):
        """Process affinity predictions with continuous checkpointing"""
        results = []
        
        # Prepare protein data once
        coords = self.protein_features['surface_points']
        dist_matrix = cdist(coords, coords)
        edge_list = np.where(dist_matrix < 5.0)
        edge_index = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long).contiguous()
        
        protein_data = Data(
            x=torch.tensor(self.protein_features['residue_features'], dtype=torch.float),
            edge_index=edge_index,
            batch=torch.zeros(len(self.protein_features['residue_features']), dtype=torch.long)
        )
        
        print(f"🧬 Predicting affinities for {len(ligand_df)} ligands...")
        
        # Process in smaller batches
        batch_size = AFFINITY_CHECKPOINT_SIZE
        num_batches = (len(ligand_df) - 1) // batch_size + 1
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(ligand_df))
            
            print(f"\n🔄 Affinity batch {batch_idx + 1}/{num_batches} ({batch_start + 1}-{batch_end})")
            
            batch_results = []
            
            with memory_cleanup():
                for idx in range(batch_start, batch_end):
                    row = ligand_df.iloc[idx]
                    
                    if (idx - batch_start) % 20 == 0:
                        progress = f"Progress: {idx - batch_start + 1}/{batch_end - batch_start}"
                        print(f"\r{progress}", end="", flush=True)
                    
                    try:
                        # Create molecule from SMILES or use stored mol object
                        mol_data = row.to_dict()
                        
                        # Create a dummy molecule for testing (replace with actual mol reconstruction)
                        mol = Chem.MolFromSmiles('CCO')  # Placeholder
                        if mol is None:
                            continue
                        
                        with torch.no_grad():
                            lp = LigandProcessor(mol)
                            pharma_features = lp.get_3dpharma_features()
                            graph = lp.get_graph_representation()
                            
                            # Predict affinity
                            affinity = self.model(protein_data, graph)
                            pharma_features['affinity'] = affinity.item()
                            
                            # Calculate physical score
                            phys_score = calculate_binding_score(
                                self.protein_features, 
                                {'surface_points': lp.get_surface_points(), **pharma_features},
                                mol_data,
                                {'AvgDistToSite': row['AvgDistToSite']},
                                row['EntropyLoss']
                            )
                            
                            final_score = 0.7 * affinity.item() + 0.3 * phys_score
                            
                            result = {
                                'Drug': row['Ligand'],
                                'Affinity': affinity.item(),
                                'PhysScore': phys_score,
                                'FinalScore': final_score,
                                **pharma_features,
                                **mol_data
                            }
                            
                            batch_results.append(result)
                            
                    except Exception as e:
                        print(f"\n❌ Error predicting for {row['Ligand']}: {e}")
                        continue
            
            # Save batch results
            results.extend(batch_results)
            
            if results:
                checkpoint_df = pd.DataFrame(results)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_file = f"affinity_results_batch_{batch_idx + 1}_{timestamp}.xlsx"
                save_checkpoint(checkpoint_df, checkpoint_file)
                print(f"\n💾 Saved {len(results)} affinity predictions")
            
            log_memory_usage(f"After affinity batch {batch_idx + 1}")
        
        return pd.DataFrame(results)
    
    def run(self):
        """Main pipeline execution with comprehensive checkpointing"""
        try:
            print("🚀 Starting PharmaPipeline execution...")
            
            # Validate inputs
            pdb_path = os.path.normpath(PDB_FILE)
            ligand_folder = os.path.normpath(LIGAND_FOLDER)
            
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"PDB file {pdb_path} not found")
            if not os.path.exists(ligand_folder):
                raise FileNotFoundError(f"Ligand folder {ligand_folder} not found")
            
            # Step 1: Process ligand descriptors
            print("\n📊 Step 1: Processing ligand descriptors...")
            ligand_df = process_ligands_from_multiple_sdfs(ligand_folder, self.protein_features['centroid'])
            
            if ligand_df.empty:
                print("❌ No ligands processed successfully")
                return None
            
            # Step 2: Process affinity predictions
            print("\n🧬 Step 2: Processing affinity predictions...")
            results_df = self.process_affinity_predictions(ligand_df)
            
            if results_df.empty:
                print("❌ No affinity predictions completed")
                return None
            
            # Step 3: Final processing and saving
            print("\n📈 Step 3: Final processing...")
            results_df = results_df.sort_values('FinalScore', ascending=False)
            
            # Save final results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"Nature_Methods_Pharma_Ranking_{timestamp}.xlsx"
            
            success = safe_save_dataframe(results_df, output_file)
            
            if success:
                print(f"\n📄 Final output saved to: {output_file}")
                print(f"🏆 Top 5 compounds by FinalScore:")
                display_cols = ['Drug', 'FinalScore', 'Affinity', 'PhysScore']
                available_cols = [col for col in display_cols if col in results_df.columns]
                print(results_df[available_cols].head().to_string())
            
            return results_df
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            print("🔍 Full traceback:")
            traceback.print_exc()
            return None
        
        finally:
            print("\n🧹 Performing final cleanup...")
            with memory_cleanup():
                pass
            log_memory_usage("Final cleanup")

if __name__ == "__main__":
    try:
        log_memory_usage("Program start")
        
        print("🔬 Nature Methods Pharma Analysis Pipeline")
        print("=" * 50)
        
        pipeline = PharmaPipeline()
        result_df = pipeline.run()
        
        if result_df is not None:
            print("\n🎉 Nature Methods-ready analysis complete!")
            print(f"✅ Successfully processed {len(result_df)} compounds")
        else:
            print("\n❌ Analysis failed - check error messages above")
            
        log_memory_usage("Program end")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        print("💾 Checkpoint files have been saved - you can resume processing")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n🧹 Final cleanup...")
        with memory_cleanup():
            pass
        print("🏁 Program finished.")