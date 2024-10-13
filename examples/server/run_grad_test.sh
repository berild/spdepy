#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=1

# echo vM24$BC
tmux new-session -d -s C1$BC "python3 grad_test.py 1 ${BC}"
# tmux new-session -d -s C2$BC "python3 grad_test.py 2 ${BC}"
# tmux new-session -d -s C3$BC "python3 grad_test.py 3 ${BC}"

# tmux new-session -d -s WM4$BC "python3 grad_test.py 4 ${BC}"
# tmux new-session -d -s WM5$BC "python3 grad_test.py 5 ${BC}"
# tmux new-session -d -s WM6$BC "python3 grad_test.py 6 ${BC}"
                                                              
# tmux new-session -d -s vWM4$BC "python3 grad_test.py 7 ${BC}"
# tmux new-session -d -s vWM5$BC "python3 grad_test.py 8 ${BC}"
# tmux new-session -d -s vWM6$BC "python3 grad_test.py 9 ${BC}"
                                                                   
tmux new-session -d -s AD11$BC "python3 grad_test.py 1 1 ${BC}"
# tmux new-session -d -s cAD21$BC "python3 grad_test.py 2 1 ${BC}"
# tmux new-session -d -s vAD31$BC "python3 grad_test.py 3 1 ${BC}"
                                                                 
# tmux new-session -d -s AD12$BC "python3 grad_test.py 1 2 ${BC}"
# tmux new-session -d -s cAD22$BC "python3 grad_test.py 2 2 ${BC}"
# tmux new-session -d -s vAD32$BC "python3 grad_test.py 3 2 ${BC}"
                                                            
# tmux new-session -d -s AD13$BC "python3 grad_test.py 1 3 ${BC}"
# tmux new-session -d -s cAD23$BC "python3 grad_test.py 2 3 ${BC}"
# tmux new-session -d -s vAD33$BC "python3 grad_test.py 3 3 ${BC}"
                                                                 
# tmux new-session -d -s AvD14$BC "python3 grad_test.py 1 4 ${BC}"
# tmux new-session -d -s cAvD24$BC "python3 grad_test.py 2 4 ${BC}"
tmux new-session -d -s vAvD34$BC "python3 grad_test.py 3 4 ${BC}"
                                                                 
# tmux new-session -d -s AvD15$BC "python3 grad_test.py 1 5 ${BC}"
# tmux new-session -d -s cAvD25$BC "python3 grad_test.py 2 5 ${BC}"
# tmux new-session -d -s vAvD35$BC "python3 grad_test.py 3 5 ${BC}"
                                                                 
# tmux new-session -d -s AvD16$BC "python3 grad_test.py 1 6 ${BC}"
# tmux new-session -d -s cAvD26$BC "python3 grad_test.py 2 6 ${BC}"
# tmux new-session -d -s vAvD36$BC "python3 grad_test.py 3 6 ${BC}"