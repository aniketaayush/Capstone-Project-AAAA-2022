export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --run_scenario 1 \
    --device cuda \
    --dataset_name action \
    --train_data_paths /content/predrnn/data/comedian/ \
    --valid_data_paths /content/predrnn/data/comedian/ \
    --save_dir /content/drive/MyDrive/Models/change_test \
    --gen_frm_dir results/predrnn/ \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 128 \
    --img_channel 3 \
    --input_length 5 \
    --total_length 10 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 5000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2000 \
    --lr 0.0001 \
    --batch_size 10 \
    --display_interval 50 \
    --test_interval 50000 \
    --train_end_no 2 \
    --test_end_no 3 \
    --snapshot_interval 1000 \
    --max_iterations 5000 
    # --pretrained_model "/content/drive/MyDrive/Models/Sep20_5000_5to5/model.ckpt-5000"