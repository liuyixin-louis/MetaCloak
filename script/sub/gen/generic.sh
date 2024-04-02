pwdd=$(cd "$(dirname "$0")";pwd)

export MODEL_PATH=$gen_model_path

# source activate $ADB_ENV_NAME; 

export WANDB_MODE=online
export CLEAN_TRAIN_DIR="$ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/set_A" 

export CLEAN_ADV_DIR="$ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/set_B"
export CLEAN_REF="$ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/set_C"
export class_name=$(cat $ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/class.txt)
# if class_name = "face", replace it with "person"
if [ "$class_name" = "face" ]; then
  class_name="person"
fi
echo $class_name
# replace blank in class_name with -
class_name=$(echo $class_name | sed "s/ /-/g")
export CLASS_DIR="$ADB_PROJECT_ROOT/prior-data/$model_name/class-$class_name"

export OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data/gen_output/$gen_exp_name/$gen_exp_hyper/${instance_name}"
mkdir -p $OUTPUT_DIR
cp -r $CLEAN_REF $OUTPUT_DIR/image_clean_ref
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise
export step_size=$(echo "scale=2; $r/10" | bc)

