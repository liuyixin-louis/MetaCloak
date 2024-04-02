
if [ "$method_select" = "metacloak" ]; then
    export method_name=MetaCloak
    export gen_file_name="metacloak.sh"
    export advance_steps=2; 
    export total_trail_num=4; 
    export interval=200; 
    export total_train_steps=1000; 
    export unroll_steps=1;
    export defense_sample_num=1;
    export ref_model_path=$gen_model_path
    export method_hyper=advance_steps-$advance_steps-total_trail_num-$total_trail_num-unroll_steps-$unroll_steps-interval-$interval-total_train_steps-$total_train_steps-$model_name
    
    if [ "$train_mode" = "gau" ]; then
        export method_hyper=$method_hyper-robust-gauK-$gauK
    fi
elif [ "$method_select" = "clean" ]; then
    export method_name=Clean
    export gen_file_name="Clean.sh"
    export method_hyper=""
else
    echo "Invalid method name $method_select"
    exit 1
fi