from context import corr_dis
import numpy as np
import matplotlib.pyplot as plt


def compute_diff_fig(noise_type, nb_avg, L, Sd, list_k, resolution, N):

    avg_fig = np.zeros(resolution)
    grating = np.linspace(1, N, N)
    for j in range(0, nb_avg):
        perturbation = noise_type(L, Sd, N)
        pos = grating + perturbation

        F = corr_dis.diffraction_figure(pos, list_k, resolution, size=0.0)
        
        avg_fig += F

    avg_fig = avg_fig / nb_avg

    return avg_fig

N = 100
resolution = N * 20
nb_avg = 10
orders = 1

Sds = [0.1, 0.3, 0.4]
Ls = [0.5, 1, 1.5]

mode = "correk"

if mode == "direct":
    S_delta = corr_dis.S_delta_direct
    noise_type = corr_dis.direct
else:
    S_delta = corr_dis.S_delta_corrective
    noise_type = corr_dis.corrective

list_kx = np.unique(
    np.concatenate(
        (
            np.linspace(0.1, orders + 0.5, resolution),
            [i for i in range(1, orders + 1)],
        )
    )
)  # Making sure we don't miss the diffraction orders

pos_orders = np.zeros(orders, dtype=int)
for i in range(orders):
    pos_orders[i] = np.where(list_kx == i + 1)[0][0]
resolution = len(list_kx)


iplot = 0
plt.figure(figsize=(5 * len(Sds), 5 * len(Ls)))
for i, Sd in enumerate(Sds):
    print(f"Outer {i+1} / {len(Sds)}")
    for j, Lc in enumerate(Ls):
        print(f"  Inner {j+1} / {len(Ls)}")

        diff_fig = compute_diff_fig(noise_type, nb_avg, Lc, Sd, list_kx, resolution, N)

        tot_n_corr = max(int(Lc*3), 1)

        stat_diff_fig  = corr_dis.analytical_average_diff_fig(S_delta, Lc, Sd, tot_n_corr, list_kx, resolution, pos_orders, N, return_type=0)

        
        plt.subplot(len(Sds), len(Ls), iplot + 1)
        plt.plot(list_kx, diff_fig, 'b', label=f"Random")
        plt.plot(list_kx, stat_diff_fig, 'r', label=f"Average")

        maxi_graph = 3 * np.max(diff_fig[: pos_orders[0] - 4])
        # plt.ylim([-0.000, maxi_graph])
        plt.legend()
        iplot += 1
plt.tight_layout()
# plt.savefig(f"figs/fig_Diff_simple_nomoy_Sds{Sds}_Ls{Ls}_N{N}_moy{nb_moyenne}.svg")

plt.show()
