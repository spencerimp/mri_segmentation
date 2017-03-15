# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# enable vim coloring in terminal
export TERM="xterm-256color"

# show screen name
if [ -n "$STY" ]; then export PS1="($STY)$ "; fi

# set openmp
export OMP_NUM_THREADS=8
export THEANO_BASE_COMPILEDIR="~/.theano"

# cudnn
export LD_LIBRARY_PATH=/home/somewhere/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/somewhere/cuda/include:$CPATH
export LIBRARY_PATH=/home/somewhere/cuda/lib64:$LD_LIBRARY_PATH

# specify Theano gpu
alias rungpu0="THEANO_FLAGS='device=gpu0' "
alias rungpu1="THEANO_FLAGS='device=gpu1' "
alias rungpu2="THEANO_FLAGS='device=gpu2' "
alias rungpu3="THEANO_FLAGS='device=gpu3' "

# libgpuarray, see http://deeplearning.net/software/libgpuarray/installation.html
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/py35env/lib64/
export LIBRARAY_PATH=$LIBRARY_PATH:~/anaonda3/envs/py35env/lib64/
export CPATH=$CPATH:~/anaconda3/envs/py35env/include/
export LIBRARY_PATH=$LIBRARY_PATH:~/anaconda3/envs/py35env/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/py35env/lib/


# if libgpuarray is installed
alias runcuda0="THEANO_FLAGS='device=cuda0' "
alias runcuda1="THEANO_FLAGS='device=cuda1' "
alias runcuda2="THEANO_FLAGS='device=cuda2' "
alias runcuda3="THEANO_FLAGS='device=cuda3' "

# specify Keras backend
alias keras_tf="KERAS_BACKEND=tensorflow KERAS_IMAGE_DIM_ORDERING=tf "
alias keras_th="KERAS_BACKEND=theano KERAS_IMAGE_DIM_ORDERING=th "

# in case your machines use a shared filesystem
alias oncluster1="export THEANO_BASE_COMPILEDIR=$HOME/.theanocluster1"
alias oncluster2="export THEANO_BASE_COMPILEDIR=$HOME/.theanocluster2"
alias oncluster3="export THEANO_BASE_COMPILEDIR=$HOME/.theanocluster3"
alias oncluster4="export THEANO_BASE_COMPILEDIR=$HOME/.theanocluster4"

# some alias and functions making my life a bit easier
killwin() {
    screen -X -S $1 quit
}

runcuda() {
    THEANO_FLAGS="base_compiledir=$THEANO_BASE_COMPILEDIR, device=cuda$1" $2 $3
}

rungpu() {
    THEANO_FLAGS="base_compiledir=$THEANO_BASE_COMPILEDIR+$1, device=gpu$1" $2 $3
}

runtf() {
     KERAS_BACKEND=tensorflow KERAS_IMAGE_DIM_ORDERING=tf CUDA_VISIBLE_DEVICES=$1 $2 $3
}
alias sls="screen -ls"
