import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_nfe_data_scaled():
    # Data from the table
    nfe = [16, 32, 64, 128, 256, 512]
    mask_cid = [9251, 9917, 9978, 9991, 9998, 9996]
    marg_ddm = [5731, 6933, 7680, 7809, 8015, 8037]
    marg_id = [3825, 7366, 9271, 9769, 9906, 9958]

    # FCD
    # mask_cid = [9.42, 6.63, 5.05, 4.09, 3.77, 3.40]
    # marg_ddm = [18.81, 13.85, 11.21, 9.73, 8.87, 8.71]
    # marg_id = [14.70, 6.37, 3.06, 2.15, 1.89, 2.03]

    # NSPDK
    # mask_cid = [11.10, 5.66, 3.55, 2.64, 2.30, 2.26]
    # marg_ddm = [47.22, 27.35, 20.23, 15.74, 13.75, 13.46]
    # marg_id = [27.86, 7.18, 1.80, 1.30, 1.53, 2.11]

    # Scale the values by dividing by 100
    critical_iterative_denoising = [val/100 for val in mask_cid]
    iterative_denoising = [val/100 for val in marg_id]
    discrete_diffusion = [val/100 for val in marg_ddm]

    color_map = cm.get_cmap("magma", 10)  # Use a vibrant colormap
    colors = color_map(np.linspace(0, 1, 10))

    # Plot each line
    plt.figure(figsize=(8, 6))

    plt.plot(nfe, discrete_diffusion, marker='^', linewidth=3.0, markersize=8.0,
             label='Discrete Diffusion (Marg.) - Baseline',  color=colors[7])
    plt.plot(nfe, iterative_denoising, marker='s', linewidth=3.0, markersize=8.0,
             label='Simple Iterative Denoising (Marg.) - Ours',  color=colors[4])
    plt.plot(nfe, critical_iterative_denoising, marker='o', linewidth=3.0, markersize=8.0,
             label='Critical Iterative Denoising (Mask) - Ours', color=colors[0])

    # Add labels and legend
    #plt.title('Validity on Zinc250k', fontsize=24)
    plt.xscale('log', base=2)
    plt.xlabel('NFE', fontsize=18)
    plt.ylabel('Validity (%)', fontsize=18)
    plt.xticks(nfe, labels=[str(x) for x in nfe], fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)

    plt.tight_layout()
    plt.show()


# Call the function to plot the data
plot_nfe_data_scaled()

def plot_scatter_properties(var1, var2, values, conditional_values, ref_values):
    prop = ('Mol. W.', 'LogP', 'QED')
    plt.figure(figsize=(8, 6))
    plt.scatter(ref_values[..., var1].cpu().numpy(),
                ref_values[..., var2].cpu().numpy(), color='green', s=10, alpha=.5,
                label='Data')
    plt.scatter(values[..., var1].cpu().numpy(),
                values[..., var2].cpu().numpy(), color='red', s=10, alpha=.5,
                label='Guided Generation')
    plt.scatter(conditional_values[..., var1].cpu().numpy(),
                conditional_values[..., var2].cpu().numpy(), color='black', s=50, marker='x',
                label='Target')
    plt.xlabel(prop[var1], fontsize=18)
    plt.ylabel(prop[var2], fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=18, markerscale=3)
    plt.tight_layout()
    plt.show()

def plot_histogram_properties(var, values, ref_values):
    prop = ('Mol. W.', 'LogP', 'QED')
    plt.figure(figsize=(8, 6))
    plt.hist(values[..., var].cpu().numpy(), bins=28, density= True, alpha=.50, color='red',
                     range=(-4, 3), label='Guided Generation')
    plt.hist(ref_values[..., var].cpu().numpy(), bins=28, density= True, alpha=.50, color='green',
                     range=(-4, 3), label='Data')
    plt.xlabel(prop[var], fontsize=18)
    # plt.xlabel('PMF', fontsize=18)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()