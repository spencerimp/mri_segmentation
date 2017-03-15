Install FSL

        python fslinstaller.py

Note that you have to execute it with Python 2.x.

Select the installation directory to ~/.local
After installation is done, edit the shell consifguration (e.g. ~/.bashrc)

export FSL_DIR="~/.local/fsl"

export PATH=$PATH:$FSL_DIR/bin

