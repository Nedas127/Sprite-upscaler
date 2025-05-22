import pandas as pd
import matplotlib.pyplot as plt
import os
import re

input_file = r'C:\Users\konte\ESRGAN\all_models_results\model_comparison_summary.txt'
output_dir = r'C:\Users\konte\ESRGAN\all_models_results'
output_file = os.path.join(output_dir, 'model_comparison_chart.png')


def parse_results_file(file_path):
    models = []
    psnr = []
    ssim = []
    mse = []
    times = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        start_reading = False

        for line in lines:
            if line.startswith('----'):
                start_reading = True
                continue
            if not start_reading or not line.strip():
                continue

            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 5:
                models.append(parts[0].strip())
                psnr.append(float(parts[1]))
                ssim.append(float(parts[2]))
                mse.append(float(parts[3]))
                times.append(float(parts[4]))

    return {
        'Model Name': models,
        'PSNR (dB)': psnr,
        'SSIM': ssim,
        'MSE': mse,
        'Time (s)': times
    }


try:
    data = parse_results_file(input_file)
    df = pd.DataFrame(data)

    df['Short Name'] = df['Model Name'].apply(lambda x: re.split(r'_|-', x)[0])
    df['Short Name'] = df['Short Name'].apply(lambda x: x[:15] + '...' if len(x) > 15 else x)

    plt.figure(figsize=(18, 12))
    plt.suptitle('Model Comparison Metrics', fontsize=16, y=1.02)

    plt.subplot(2, 2, 1)
    plt.barh(df['Short Name'], df['PSNR (dB)'], color='skyblue')
    plt.xlabel('PSNR (dB)')
    plt.title('Peak Signal-to-Noise Ratio (Higher is better)')
    plt.gca().invert_yaxis()

    plt.subplot(2, 2, 2)
    plt.barh(df['Short Name'], df['SSIM'], color='lightgreen')
    plt.xlabel('SSIM')
    plt.title('Structural Similarity (Higher is better)')
    plt.gca().invert_yaxis()

    plt.subplot(2, 2, 3)
    plt.barh(df['Short Name'], df['MSE'], color='salmon')
    plt.xlabel('MSE')
    plt.title('Mean Squared Error (Lower is better)')
    plt.gca().invert_yaxis()

    plt.subplot(2, 2, 4)
    plt.barh(df['Short Name'], df['Time (s)'], color='gold')
    plt.xlabel('Time (seconds)')
    plt.title('Processing Time (Lower is better)')
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f'Diagrama sėkmingai išsaugota: {output_file}')

except FileNotFoundError:
    print(f'Klaida: Failas nerastas - {input_file}')
except Exception as e:
    print(f'Klaida apdorojant duomenis: {str(e)}')