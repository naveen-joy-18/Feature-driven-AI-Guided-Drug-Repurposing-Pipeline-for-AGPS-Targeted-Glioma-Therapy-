# Feature-driven-AI-Guided-Drug-Repurposing-Pipeline-for-AGPS-Targeted-Glioma-Therapy-
We present an AI-driven pipeline for repurposable cancer drug discovery. Applied to glioma, we identified AGPS as a target, screened 576k compounds, and found F2881-0267 with strong stability and drug-like features. The framework generalizes across cancers.
Cancer remains among the most aggressive and treatment-resistant diseases, with drug discovery pipelines often failing due to bottlenecks in target identification, hit prioritization, and binding free energy estimation.

This repository presents a feature-driven AI-integrated pipeline designed for:

Systematic identification of undercharacterized druggable targets from biomedical literature.

Screening and ranking of compounds against targets using graph neural networks (GNNs), docking, and hybrid deep learning models.

Frame-wise binding free energy prediction from molecular dynamics (MD) trajectories using 3D CNN + MLP–based models.

Interactive assistants for literature mining, protein scoring, and glioma-specific Q&A.

The framework is demonstrated on glioma, with a focus on Alkylglycerone phosphate synthase (AGPS), a key enzyme in tumor metabolism and progression.
Installation
1. Clone the repository
git clone https://github.com/<your-username>/glioma-drug-discovery.git
cd glioma-drug-discovery

2. Create environment

We recommend using Python 3.10+.

conda create -n glioma-pipeline python=3.10 -y
conda activate glioma-pipeline

3. Install dependencies
pip install -r requirements.txt


Key packages:

torch, transformers, sentence-transformers

rdkit, biopython, prolif, mdtraj

faiss, pandas, numpy, scikit-learn

gradio, matplotlib, seaborn, plotly

🚀 Components
1. Binding Free Energy Model (Binding free energy model.py)

Implements 3D CNN (for voxelized protein-ligand grids).

Integrates molecular fingerprints (ProLIF) and physics descriptors (electrostatics, VdW, solvation).

Predicts ΔG per MD frame, trained with regression loss.

Includes entropy correction and solvation estimation.

2. Prediction Script (prediction.py)

Command-line interface for batch prediction from MD trajectories.

Usage:

python prediction.py \
  --topology_file complex.gro \
  --trajectory_dir traj.xtc \
  --model_path hybrid_model.pth \
  --frame_limit 100 \
  --temperature 298 \
  --excel_output binding_energies.xlsx \
  --plot_output binding_plot.png


Outputs:

Excel file of ΔG per frame.

Plot showing ΔG trajectory with mean ± SD.

Boltzmann-weighted ΔG estimate.

3. Language Model Assistant (language model.py)

Loads abstracts from abstracts.xlsx.

Embeds them using Sentence-BERT.

Provides a Glioma Research Assistant (Flan-T5-XXL) for question answering about AGPS and glioma research.
