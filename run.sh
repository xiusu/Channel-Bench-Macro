#!/usr/bin/env bash
#--gres=gpu:1
part=$1
jobname=$2
shift
shift
GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $jobname --partition=$part -x BJ-IDC1-10-10-16-[46,83,85,51,53,60,61,86,88] -n1 --gres=gpu:1 --ntasks-per-node=1 "$@"

