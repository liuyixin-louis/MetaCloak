# for clean case simply do copying 
dir_of_this_file="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir_of_this_file/generic.sh

###### the following are method-related variables ######
round=final
INSTANCE_DIR_CHECK="$OUTPUT_DIR/noise-ckpt/${round}"
eps=$(echo "scale=3; $r/255" | bc)

source activate $ADB_ENV_NAME
cd $ADB_PROJECT_ROOT/robust_facecloak
if [ ! -d "$OUTPUT_DIR/noise-ckpt/" ]; then 
  {
    mkdir $OUTPUT_DIR/noise-ckpt/
  }
fi
# if [ ! -d "$OUTPUT_DIR/noise-ckpt/${round}"]; then
if [ ! -d "$INSTANCE_DIR_CHECK" ]; then
{
    mkdir $INSTANCE_DIR_CHECK
}
fi
# move all files inside CLEAN_TRAIN_DIR to 
cp -r $CLEAN_TRAIN_DIR/* $INSTANCE_DIR_CHECK