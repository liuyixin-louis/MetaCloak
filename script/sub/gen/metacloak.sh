source activate $ADB_ENV_NAME;
dir_of_this_file="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $dir_of_this_file/generic.sh

###### the following are method-related variables ######
alg_file_name="metacloak"
round=final
INSTANCE_DIR_CHECK="$OUTPUT_DIR/noise-ckpt/${round}"

# defense_sample_num
if [ -z "$defense_sample_num" ]; then 
  defense_sample_num=1
fi

cd $ADB_PROJECT_ROOT/robust_facecloak
# skip if noise exists 
if [ ! -d "$INSTANCE_DIR_CHECK" ]; then 
  {
    command="""python3 attacks/algs/$alg_file_name.py --instance_name $instance_name --dataset_name $dataset_name \
    --total_train_steps $total_train_steps \
    --wandb_entity_name $wandb_entity_name \
    --seed $seed \
    --interval $interval \
    --advance_steps $advance_steps \
    --unroll_steps $unroll_steps \
    --total_trail_num $total_trail_num \
    --exp_name $gen_exp_name \
    --exp_hyper $gen_exp_hyper \
    --pretrained_model_name_or_path=$ref_model_path  \
    --enable_xformers_memory_efficient_attention \
    --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
    --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
    --instance_prompt='a photo of $gen_prompt $class_name' \
    --class_data_dir=$CLASS_DIR \
    --num_class_images=200 \
    --class_prompt='a photo of $class_name' \
    --output_dir=$OUTPUT_DIR \
    --center_crop \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --resolution=512 \
    --train_text_encoder \
    --train_batch_size=1 \
    --checkpointing_iterations=10 \
    --learning_rate=5e-7 \
    --defense_pgd_radius=$r \
    --defense_pgd_step_size=$step_size \
    --defense_pgd_step_num=6 \
    --defense_sample_num=$defense_sample_num \
    --defense_pgd_ascending \
    --attack_pgd_radius=0 \
    --attack_pgd_step_size=1 \
    --attack_pgd_step_num=3 \
    --mixed_precision=fp16"""
    
    if [ "$train_mode" = "gau" ]; then
      command="$command --transform_gau --gau_kernel_size $gauK --transform_hflip "
    fi

    echo $command
    eval $command
  }
else
  echo "instance dir exists"
fi; 