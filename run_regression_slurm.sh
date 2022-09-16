#!/bin/bash

# Arg parsing
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

programname=$0
function usage {
    echo ""
    echo "Run NP regression experiments."
    echo ""
    echo "usage: ${programname} --exp string --mode string --model string --expid string "
    echo ""
    echo "  --exp string            Which experiment to run: [gp | celeba | eminst]"
    echo "  --mode string           Usage: train/eval"
    echo "  --model string          Model to use"
    echo "  --expid string          Unique name to give experiment (optional)"
    echo ""
}

function die {
    printf "Script failed: %s\n\n" "${1}"
    exit 1
}

if [[ -z "${exp}" ]]; then
    usage
    die "Missing parameter --exp"
elif [[ -z "${mode}" ]]; then
    usage
    die "Missing parameter --mode"
elif [[ -z "${model}" ]]; then
    usage
    die "Missing parameter --model"
fi

base_results_dir="/share/kuleshov/yzs2/TNP-pytorch/regression/results"
results_dir="${base_results_dir}"  # Will be overwritten in cases below

# Start building command line exports
export_str="ALL,mode=${mode},model=${model},expid=${expid}"

# GP:
if [[ "${exp}" == "gp" ]]; then
  if [[ -z "${max_num_pts}" ]]; then
    max_num_pts=50
    echo "Missing GP parameter --max_num_pts. Defaulting to ${max_num_pts}."
  fi
  if [[ -z "${min_num_ctx}" ]]; then
    min_num_ctx=3
    echo "Missing GP parameter --min_num_ctx. Defaulting to ${min_num_ctx}."
  fi
  if [[ -z "${min_num_tar}" ]]; then
    min_num_tar=3
    echo "Missing GP parameter --min_num_tar. Defaulting to ${min_num_tar}."
  fi
  if [[ -z "${eval_kernel}" ]]; then
    eval_kernel="rbf"
    echo "Missing GP parameter --eval_kernel. Defaulting to ${eval_kernel}."
  fi
  export_str="${export_str},max_num_pts=${max_num_pts},min_num_ctx=${min_num_ctx},min_num_tar=${min_num_tar},eval_kernel=${eval_kernel}"
  if [[ -n "${eval_logfile}" ]]; then
    export_str="${export_str},eval_logfile=${eval_logfile}"
  fi

  results_dir="${base_results_dir}/${exp}_maxpts-${max_num_pts}_minctx-${min_num_ctx}_mintar-${min_num_tar}/${model}"
  # Build expid
  if [[ -z "${expid}" ]]; then
      echo "Missing parameter --expid; expid will be created automatically."
      last_exp=-1
      if [ -d "${results_dir}" ] && [ "$(ls -A "${results_dir}")" ]; then  # Check if dir is exists and is not empty
        for dir in "${results_dir}/"*; do  # Loop over experiment versions and find most recent "v[expid]"
          if [[ ${dir} = "${results_dir}/v"* ]]; then  # Only look at dirs that mathc v[expid]
            dir=${dir%*/}      # remove the trailing "/"
            current_exp="${dir//$results_dir\/v/}"
            if [[ $current_exp -gt $last_exp ]]; then
              last_exp=$current_exp
            fi
          fi
        done
      fi
      expid="v$((last_exp+1))"
  fi
  export_str="${export_str},expid=${expid}"
fi

# Build Job name / make log dir
job_name="${exp}_${model}_${mode}_${expid}"
log_dir="${results_dir}/${expid}/logs"
mkdir -p "${log_dir}"

echo "Scheduling Job: ${job_name}"
sbatch \
  --job-name="${job_name}" \
  --output="${log_dir}/${mode}_%j.out" \
  --error="${log_dir}/${mode}_%j.err" \
  --export="${export_str}" \
  "run_${exp}.sh"
