#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=3

tmux new-session -d -s aC$BC "python3 seperable_spatial_temporal.py ${BC}"

# tmux new-session -d -s aWM$BC "python3 whittle_matern.py ${BC}"

# tmux new-session -d -s avWM$BC "python3 var_whittle_matern.py ${BC}"

tmux new-session -d -s aAD$BC "python3 advection_diffusion.py ${BC}"

tmux new-session -d -s avAD$BC "python3 var_advection_diffusion.py ${BC}"