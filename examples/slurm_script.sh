#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=genetic_algo
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=11:50:00
#SBATCH --mail-user=your@email.here
#SBATCH --mail-type=ALL

set -x

module load cmake/3.23.1
module load gcc/9.3.0
module load StdEnv/2020
module load osrm-backend/5.26.0
module load fmt/7.0.3
module load geos/3.10.2
module load libspatialindex/1.8.5
module load python/3.10.2
module load scipy-stack
module load boost/1.72.0
module load tbb/2020.2
module load proj/9.0.1

RUNPATH=/home/dmurray/cycling-network-design/
cd $RUNPATH
source bin/activate

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6384
ray_head_node_ip=$head_node_ip:$port
export ray_head_node_ip
echo "IP Head: $ray_head_node_ip"
echo "SLURM_CPUS_PER_TASK ${SLURM_CPUS_PER_TASK}"
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --port=$port --block &


# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "worker_num ${worker_num}"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting NODE $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ray_head_node_ip" --block &
    sleep 5
done

python code/genetic_algorithm.py --data_folder data --output_folder test_run3 -d --n_workers 1 --n_pop 5 --value_of_time 15 --unreachable_trip_cost 25 --runtime_limit 3600 --cxpb 0.95 --mutpb 0.05 --n_elites_prop 0.05 --tourn_size 20 --mate probabilisticGeneCrossover --mutation_rate_type constant 