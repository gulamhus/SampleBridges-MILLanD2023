#!/usr/bin/env bash
# by Oleksii Bashkanov

function usage() {
    echo "Usage: bash $0 -v /home/env -s sweeps/sw_bs_lr.yaml --gpus 0,1"
    echo "Automatically populates agents in separate screens and GPUs."
    echo ""
    echo "Arguments:"
    echo -e "  -v --venv\t\tPath to the virtual enviroment with wandb and tensorflow installed."
    echo -e "  -s --sweep\t\tPath to the sweep."
    echo -e "  -g --gpus\t\tGPUs to use. Can be comma-separated like: 0,1"
    exit 1
}

#parse arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -v|--venv)
      VENV="$2"
      ;;
    -s|--sweep)
      SWEEP_PATH="$2"
      ;;
    -g|--gpus)
      GPUS="$2"
      ;;
    -h | --help | * )
    usage;;
  esac
  shift
  shift
done

# as string
AVAILGPUSLIST=$(nvidia-smi -L | awk -F": " '{print $1}' | awk '{print $2}')
# as list
#AVAILGPUSLIST=($(nvidia-smi -L | awk -F": " '{print $1}' | awk '{print $2}'))

function list_include_item {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
  else
    result=1
  fi
  return $result
}

echo "Found following GPUs:"
nvidia-smi -L
echo "Checking GPUs..."

# iterate over gpus
IFS=',' read -ra GPULIST <<< "$GPUS"
for i in "${GPULIST[@]}";do
  if ! list_include_item "$AVAILGPUSLIST" "$i"; then
    echo "Error: requested GPU ${i} is not found. Please use only available GPUs!"; exit 1;
  fi
done

# either run from enviroment with wandb installed or go to this enviromnet
source $(realpath ${VENV})/bin/activate
AGENT=$(wandb sweep $SWEEP_PATH 2> >(awk 'BEGIN { FS=": " } END{ print $3 }'))
echo "Generated agent for ${SWEEP_PATH} is > \"${AGENT}\""

for i in "${GPULIST[@]}";do
  screen_name=agent_$i
  # suppress output
  screen -S $screen_name -X quit >/dev/null
  echo "Running separate screen ${screen_name} on GPU ${i}... To attach this screen run: screen -r ${screen_name}"
  screen -d -m -S $screen_name bash -c "cd ${PWD}; export CUDA_VISIBLE_DEVICES=${i}; ${AGENT}"
done