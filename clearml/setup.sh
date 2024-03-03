#!/bin/bash
echo "Running setup script."

# activate env
source /root/miniconda3/bin/activate
conda activate safe-slac

# go to the repo directory
cd $CLEARML_GIT_ROOT

# install deps
pip install -r requirements.txt
pip install clearml
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e /home/realworldrl_suite
# now we need to tell clearml to use the python from our poetry env
# this is in the general case (we use the system python above, so we could
# have just hardcoded this as well)
export python_path="/root/miniconda3/envs/safe-slac/bin/python"
cat > $CLEARML_CUSTOM_BUILD_OUTPUT << EOL
{
  "binary": "$python_path",
  "entry_point": "$CLEARML_GIT_ROOT/$CLEARML_TASK_SCRIPT_ENTRY",
  "working_dir": "$CLEARML_GIT_ROOT/$CLEARML_TASK_WORKING_DIR"
}
EOL
