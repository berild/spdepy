#!/bin/sh
source ../../../../pyenvs/pacenv/bin/activate

BC=1

tmux new-session -d -s RadT$BC "python3 results.py 0 ${BC}"

tmux new-session -d -s RadC$BC "python3 results.py 1 ${BC}"

tmux new-session -d -s RadAD$BC "python3 results.py 2 ${BC}"

tmux new-session -d -s RadvAvD$BC "python3 results.py 3 ${BC}"
