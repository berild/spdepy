#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=3

# tmux new-session -d -s C$BC "python3 seperable_spatial_temporal.py ${BC}"

tmux new-session -d -s WM$BC "python3 whittle_matern.py ${BC}"

tmux new-session -d -s vWM$BC "python3 var_whittle_matern.py ${BC}"

# tmux new-session -d -s AD$BC "python3 advection_diffusion.py ${BC}"

# tmux new-session -d -s AvD$BC "python3 advection_var_diffusion.py ${BC}"

# tmux new-session -d -s cAD$BC "python3 cov_advection_diffusion.py ${BC}"

# tmux new-session -d -s cAvD$BC "python3 cov_advection_var_diffusion.py ${BC}"

# tmux new-session -d -s vAD$BC "python3 var_advection_diffusion.py ${BC}"

# tmux new-session -d -s vAvD$BC "python3 var_advection_var_diffusion.py ${BC}"