import simCell as fa

nominal = fa.Cell.build_from_pngs("/Users/federicocantarelli/Documents/tesinuova/slices")

real = fa.Cell.build_from_pngs("/Users/federicocantarelli/Documents/tesinuova/slices")

real.simulate_real()
real.save_in_gif("/Users/federicocantarelli/Documents/tesinuova/slices_reali")

real.get_deviation_maps(nominal, which = "hausdorff")
real.save_to_cmap("/Users/federicocantarelli/Documents/tesinuova/cmap_reali")


