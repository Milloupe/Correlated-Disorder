from context import corr_dis
import numpy as np
import matplotlib.pyplot as plt


# def compute_diff_fig(noise_type, nb_avg, Lx, Lz, Sdx, Sdz, list_kx, k, resolution, N):

#     avg_fig = np.zeros(resolution)
#     grating = np.linspace(1, N, N)
#     for j in range(0, nb_avg):
#         print(j,"/",nb_avg)
#         perturbation = noise_type(L, Sd, N)
#         pos = grating + perturbation

#         F = corr_dis.diffraction_figure(pos, list_k, resolution, size=0.0)

#         avg_fig += F

#     avg_fig = avg_fig / nb_avg

#     return avg_fig

N = 100
resolution = N * 20
nb_avg = 10
orders = 1
k = 3

# Sds = [4.0]
# Ls = [40]

mode = "direct"

if mode == "direct":
    S_delta = corr_dis.S_delta_direct
    noise_type = corr_dis.direct
else:
    S_delta = corr_dis.S_delta_corrective
    noise_type = corr_dis.corrective

list_kx = np.unique(
    np.concatenate(
        (
            np.linspace(-orders - 0.5, orders + 0.5, resolution),
            [i for i in range(-orders, orders + 1)],
        )
    )
)  # Making sure we don't miss the diffraction orders

pos_orders = np.zeros(2 * orders + 1, dtype=int)
for i in range(-orders, orders + 1):
    pos_orders[i] = np.where(list_kx == i)[0][0]
resolution = len(list_kx)

Sdx = 0.0
Lcx = 0.5
Sdz = 0.2
Lczs = [2.5, 1.5, 0.75]
tot_n_corr = 10

# diff_fig = compute_diff_fig(noise_type, nb_avg, Lc, Sd, list_kx, resolution, N)

# tot_n_corr = max(int(Lc*3), 1)

for Lcz in Lczs:
    stat_diff_fig = corr_dis.analytical_average_diff_fig_z(
        S_delta,
        Lcx,
        Lcz,
        Sdx,
        Sdz,
        tot_n_corr,
        list_kx,
        k,
        resolution,
        pos_orders,
        N,
        return_type=0,
    )

    # plt.plot(list_kx, diff_fig, 'b', label=f"Random")
    plt.plot(list_kx, stat_diff_fig, label=f"z average: Lc={Lcz}, Sd={Sdz}")

# maxi_graph = 3 * np.max(diff_fig[: pos_orders[0] - 4])
# plt.ylim([-0.000, 0.04])
plt.legend()
plt.tight_layout()
# plt.savefig(f"figs/fig_Diff_simple_nomoy_Sds{Sds}_Ls{Ls}_N{N}_moy{nb_moyenne}.svg")

plt.show()
