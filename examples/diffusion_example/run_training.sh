#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=3

tmux new-session -d -s dC$BC "python3 seperable_spatial_temporal.py ${BC}"

# tmux new-session -d -s dWM$BC "python3 whittle_matern.py ${BC}"

# tmux new-session -d -s dvWM$BC "python3 var_whittle_matern.py ${BC}"

tmux new-session -d -s dAD$BC "python3 advection_diffusion.py ${BC}"

tmux new-session -d -s dAvD$BC "python3 advection_var_diffusion.py ${BC}"
