#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=3

# tmux new-session -d -s C$BC "python3 seperable_spatial_temporal.py ${BC}"

# tmux new-session -d -s vAvD$BC "python3 var_advection_var_diffusion.py ${BC}"

tmux new-session -d -s AD$BC "python3 advection_diffusion.py ${BC}"

tmux new-session -d -s cAvD$BC "python3 cov_advection_var_diffusion.py ${BC}"