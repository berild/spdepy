#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

#tmux new-session -d -s WM1 'python3 whittle_matern.py 1'
#tmux new-session -d -s WM2 'python3 whittle_matern.py 2'
#tmux new-session -d -s WM3 'python3 whittle_matern.py 3'

#tmux new-session -d -s AD1 'python3 advection_diffusion.py 1'
#tmux new-session -d -s AD2 'python3 advection_diffusion.py 2'
tmux new-session -d -s AD3 'python3 advection_diffusion.py 3'

#tmux new-session -d -s AvD1 'python3 advection_var_diffusion.py 1'
#tmux new-session -d -s AvD2 'python3 advection_var_diffusion.py 2'
tmux new-session -d -s AvD3 'python3 advection_var_diffusion.py 3'

#tmux new-session -d -s cAD1 'python3 cov_advection_diffusion.py 1'
#tmux new-session -d -s cAD2 'python3 cov_advection_diffusion.py 2'
tmux new-session -d -s cAD3 'python3 cov_advection_diffusion.py 3'

#tmux new-session -d -s cAvD1 'python3 cov_advection_var_diffusion.py 1'
#tmux new-session -d -s cAvD2 'python3 cov_advection_var_diffusion.py 2'
tmux new-session -d -s cAvD3 'python3 cov_advection_var_diffusion.py 3'

#tmux new-session -d -s vAD1 'python3 var_advection_diffusion.py 1'
#tmux new-session -d -s vAD2 'python3 var_advection_diffusion.py 2'
tmux new-session -d -s vAD3 'python3 var_advection_diffusion.py 3'

#tmux new-session -d -s vAvD1 'python3 var_advection_var_diffusion.py 1'
#tmux new-session -d -s vAvD2 'python3 var_advection_var_diffusion.py 2'
tmux new-session -d -s vAvD3 'python3 var_advection_var_diffusion.py 3'