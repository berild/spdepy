#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=3

tmux new-session -d -s adC$BC "python3 seperable_spatial_temporal.py ${BC}"

# tmux new-session -d -s adWM$BC "python3 whittle_matern.py ${BC}"

# tmux new-session -d -s advWM$BC "python3 var_whittle_matern.py ${BC}"

tmux new-session -d -s adAD$BC "python3 advection_diffusion.py ${BC}"

tmux new-session -d -s advAvD$BC "python3 var_advection_var_diffusion.py ${BC}"