import matplotlib

matplotlib.use('TkAgg')
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

model_name = "4x_foolhardy_Remacri"
base_dir = r"C:\Users\konte\ESRGAN\all_models_results"
model_folder = os.path.join(base_dir, model_name + "_results")
json_file = os.path.join(model_folder, model_name + "_metrics.json")

with open(json_file, 'r') as f:
    data = json.load(f)

if "_average" in data:
    data.pop("_average")

df = pd.DataFrame(data).T
df.index = df.index.astype(str)
df = df.sort_index()

colors = {
    'psnr': '#FF8000',  # Tamsiai oranžinė
    'ssim': '#00A64C',  # Tamsiai žalia
    'mse': '#0066CC',  # Tamsiai mėlyna
    'background': '#F5F5F5',  # Šviesiai pilka
    'grid': '#E0E0E0'  # Šviesesnė pilka tinkleliui
}

plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

# Pirmas subplotas: PSNR
ax1 = plt.subplot(gs[0])
ax1.plot(range(len(df.index)), df["psnr"], color=colors['psnr'], marker='o', linewidth=2, markersize=6)
ax1.set_ylabel("PSNR", fontsize=12, fontweight='bold', color=colors['psnr'])
ax1.tick_params(axis='y', labelcolor=colors['psnr'])
ax1.set_title(f"{model_name} metrikų diagrama", fontsize=14, fontweight='bold', pad=20)
ax1.set_facecolor(colors['background'])
ax1.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Rodyti x ašies žymes
ax1.set_xticks(range(len(df.index)))
ax1.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
# Pridėti reikšmes virš taškų
for i, v in enumerate(df["psnr"]):
    ax1.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                 xytext=(0, 5), ha='center', fontsize=8, color=colors['psnr'])

# Antras subplotas: SSIM
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(range(len(df.index)), df["ssim"], color=colors['ssim'], marker='s', linewidth=2, markersize=6)
ax2.set_ylabel("SSIM", fontsize=12, fontweight='bold', color=colors['ssim'])
ax2.tick_params(axis='y', labelcolor=colors['ssim'])
ax2.set_facecolor(colors['background'])
ax2.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Rodyti x ašies žymes
ax2.set_xticks(range(len(df.index)))
ax2.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
# Pridėti reikšmes virš taškų
for i, v in enumerate(df["ssim"]):
    ax2.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                 xytext=(0, 5), ha='center', fontsize=8, color=colors['ssim'])

# Trečias subplotas: MSE
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.plot(range(len(df.index)), df["mse"], color=colors['mse'], marker='^', linewidth=2, markersize=6)
ax3.set_ylabel("MSE", fontsize=12, fontweight='bold', color=colors['mse'])
ax3.set_xlabel("Vaizdo ID", fontsize=12, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=colors['mse'])
ax3.set_facecolor(colors['background'])
ax3.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
# Rodyti x ašies žymes
ax3.set_xticks(range(len(df.index)))
ax3.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
# Pridėti reikšmes virš taškų tūkstantųjų tikslumu
for i, v in enumerate(df["mse"]):
    ax3.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                 xytext=(0, 5), ha='center', fontsize=8, color=colors['mse'])

for ax in [ax1, ax2, ax3]:
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)

plt.tight_layout(pad=2.0)

output_path = os.path.join(model_folder, "metrics_diagram.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()