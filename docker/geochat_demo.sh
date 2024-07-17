#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ] ; do SOURCE="$(readlink "$SOURCE")"; done
SCRIPTDIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

set -e

. ~/miniconda3/bin/activate
conda activate ${APPLICATION_VENV}

cd $SOURCE_ROOT

python -c "import torch;print(torch.cuda.get_device_name(0))"
python "${SOURCE_ROOT}/geochat_demo.py" --model-path  "${WEIGHTS_ROOT}"
