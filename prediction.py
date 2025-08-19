# prediction.py (Corrected)
# Changes:
# - Removed hardcoded Windows paths; use relative or configurable paths.
# - Increased frame_limit to 100 for better sampling.
# - Added command-line args for flexibility.
# - Removed dependency on 'correct_code'; assume imports from binding_free_energy_model.py (rename if needed).
# - Cleaned up sign convention comment.
# - Added entropy correction reference (though not implemented).
# - Improved error handling and outputs.
# - Ensured no temp files leftover.

import torch
import numpy as np
import mdtraj as md
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from Binding_free_energy_model import HybridBindingModel, MDTrajectoryProcessor  # Assuming the other file is renamed accordingly

# Settings
R = 0.001987  # kcal/mol·K

def boltzmann_binding_free_energy(energies, temperature=298):
    RT = R * temperature
    energies = np.array(energies)
    min_energy = np.min(energies)  # for numerical stability
    exp_weights = np.exp(-(energies - min_energy) / RT)
    partition = np.mean(exp_weights)
    delta_g = -RT * np.log(partition) + min_energy
    return delta_g

def main(args):
    print("[INFO] Loading model...")
    model = HybridBindingModel(fp_dim=128)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print("[INFO] Preparing trajectory...")
    processor = MDTrajectoryProcessor(args.topology_file, args.trajectory_dir)

    energies = []
    failed_frames = []

    print(f"[INFO] Running prediction on {args.frame_limit} frames...")
    for frame_idx in range(args.frame_limit):
        try:
            sample = processor.process_frame(frame_idx)
            with torch.no_grad():
                pred = model(
                    sample['voxel'].unsqueeze(0),
                    sample['fingerprint'].unsqueeze(0),
                    sample['physics'].unsqueeze(0)
                )
                dg = -pred.item()  # Adjust sign based on model output convention (more negative ΔG = tighter binding)
                print(f"[Frame {frame_idx}] Model-predicted ΔG: {dg:.2f} kcal/mol")
                energies.append(dg)
        except Exception as e:
            print(f"[WARN] Skipping frame {frame_idx}: {e}")
            failed_frames.append(frame_idx)

    if len(energies) > 0:
        delta_g = boltzmann_binding_free_energy(energies, temperature=args.temperature)
        std_dev = np.std(energies)

        # Save Excel
        print("[INFO] Saving results to Excel...")
        df = pd.DataFrame({
            "Frame": list(range(len(energies))),
            "Binding_Free_Energy (kcal/mol)": energies
        })
        df.to_excel(args.excel_output, index=False)

        # Plot
        print("[INFO] Generating plot...")
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=list(range(len(energies))), y=energies, marker="o", label="Predicted ΔG per Frame")
        plt.axhline(delta_g, color="red", linestyle="--", label=f"Boltzmann-weighted ΔG = {delta_g:.2f} kcal/mol")
        plt.fill_between(range(len(energies)), delta_g - std_dev, delta_g + std_dev,
                         color="red", alpha=0.2, label=f"±1 SD = {std_dev:.2f}")
        plt.xlabel("Frame Index")
        plt.ylabel("Binding Free Energy (kcal/mol)")
        plt.title("Frame-wise Binding Free Energy Prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_output, dpi=300)
        plt.close()  # Non-blocking

        print("\n================ FINAL RESULT ================")
        print(f"Boltzmann-weighted Binding Free Energy: {delta_g:.2f} kcal/mol")
        print(f"Standard Deviation: {std_dev:.2f} kcal/mol")
        print(f"Processed {len(energies)} frames (Failed: {len(failed_frames)})")
        print(f"Excel saved as: {args.excel_output}")
        print(f"Plot saved as: {args.plot_output}")
        print("==============================================")
        if failed_frames:
            print(f"Failed frames: {failed_frames}")
    else:
        print("[ERROR] No valid frames were processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Binding Free Energy Prediction")
    parser.add_argument('--topology_file', type=str, required=True, help="Topology file (e.g., correct_full_system.gro)")
    parser.add_argument('--trajectory_dir', type=str, required=True, help="Trajectory directory")
    parser.add_argument('--model_path', type=str, default="hybrid_model.pth", help="Model path")
    parser.add_argument('--frame_limit', type=int, default=100, help="Number of frames to process")
    parser.add_argument('--temperature', type=float, default=298, help="Temperature in Kelvin")
    parser.add_argument('--excel_output', type=str, default="binding_energies_per_frame.xlsx", help="Excel output file")
    parser.add_argument('--plot_output', type=str, default="binding_energy_plot.png", help="Plot output file")
    args = parser.parse_args()
    main(args)