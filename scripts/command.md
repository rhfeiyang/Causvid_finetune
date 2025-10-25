# finetune
# all data
python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_ode_finetune.sh -o1 dataset_path /home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample -o2 metadata_path /home/coder/code/video_sketch/data/custom_sketch0618/metadata_detailed.csv -o3 lr 2e-6 -o4 num_epochs 30 --gpu_num 7 --time 99:00:00 --mem 10

# no animal
python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_ode_finetune.sh -o1 dataset_path /home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample -o2 metadata_path /home/coder/code/video_sketch/data/custom_sketch0618/metadata_detailed_noanimal.csv -o3 lr 2e-6 -o4 num_epochs 15 --gpu_num 7 --time 99:00:00 --mem 10

# no animal - distill data
python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_ode_finetune.sh -o1 dataset_path /home/coder/code/video_sketch/data/sketch_distill/data_combine -o2 metadata_path /home/coder/code/video_sketch/data/sketch_distill/metadata_combine.csv -o3 lr 2e-6 -o4 num_epochs 10 --gpu_num 7 --time 99:00:00 --mem 10

# no animal - combine_filtered data
python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_ode_finetune.sh -o1 dataset_path /home/coder/code/video_sketch/data/sketch_distill/data_combine_filtered_empty0 -o2 metadata_path /home/coder/code/video_sketch/data/sketch_distill/metadata_combine_filtered_empty0.csv -o3 lr 2e-6 -o4 num_epochs 5 --gpu_num 7 --time 99:00:00 --mem 10

# no animal - combine_filtered data 50 sample
python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_ode_finetune.sh -o1 dataset_path /home/coder/code/video_sketch/data/sketch_distill/data_combine_filtered_empty1 -o2 metadata_path /home/coder/code/video_sketch/data/sketch_distill/metadata_combine_filtered_empty1_50.csv -o3 lr 2e-6 -o4 num_epochs 10 --gpu_num 7 --time 99:00:00 --mem 10

---

# set ckpts /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep15_f81_480x832_custom_sketch0618_tr_metadata_detailed_noanimal_slurm315

set ckpts /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep10_f81_480x832_sketch_distill_data__metadata_combine_filtered_empty1_50_slurm169

# inference finetune
for ckpt in $ckpts
    for epoch in (seq -f "%03g" 9 -1 9)
        
        set ckpt_folder "$ckpt/checkpoint_epoch_$epoch"

        # python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/batch_inference_causvid.sh -o1 ckpt_folder $ckpt_folder -o2 prompts_file /home/coder/code/video_sketch/data/custom_sketch0618/validation_prompts_detailed.txt -o3 seed 0 -o4 num_rollout 3 -o5 num_overlap_frames 3 -o6 background_image /home/coder/code/video_sketch/data/blank_canvas.jpg

        # python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/batch_inference_causvid.sh -o1 ckpt_folder $ckpt_folder -o2 prompts_file /home/coder/code/video_sketch/data/caption_gen/sketch_captions_quickdraw.txt -o3 seed 0 -o4 num_rollout 3 -o5 num_overlap_frames 3

        python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/batch_inference_causvid.sh -o1 ckpt_folder $ckpt_folder -o2 prompts_file /home/coder/code/video_sketch/data/sketch_distill/quickdraw_last50.txt -o3 seed 0 -o4 num_rollout 1 3 -o5 num_overlap_frames 3

        # python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/batch_inference_causvid.sh -o1 ckpt_folder $ckpt_folder -o2 prompts_file /home/coder/code/video_sketch/data/custom_sketch0618/validation_prompts_detailed.txt -o3 seed 0 -o4 num_rollout 1 3 -o5 num_overlap_frames 3 -o6 background_image /home/coder/code/video_sketch/data/first_frames/circle.jpg

    end
end



# upload wandb
python upload_checkpoint_results_to_wandb.py /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr1e-5_ep10_f81_480x832_custom_sketch0618_tr_metadata_detailed_slurm258

python upload_checkpoint_results_to_wandb.py /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep10_f81_480x832_custom_sketch0618_tr_metadata_detailed_slurm254

python upload_checkpoint_results_to_wandb.py /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep30_f81_480x832_custom_sketch0618_tr_metadata_detailed_slurm4


python upload_checkpoint_results_to_wandb.py /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep5_f81_480x832_sketch_distill_data__metadata_combine_filtered_empty0_slurm165

python upload_checkpoint_results_to_wandb.py /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep5_f81_480x832_sketch_distill_data__metadata_combine_filtered_empty1_slurm160

python upload_checkpoint_results_to_wandb.py /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_finetune_lr2e-6_ep10_f81_480x832_sketch_distill_data__metadata_combine_filtered_empty1_50_slurm169 --epochs 9
---

# distillation finetune
# python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_distillation_finetune.sh -o1 lr 1e-5 -o2 num_epochs 100 -o3 resume_from /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333 --gpu_num 7 --time 99:00:00 --mem 10

python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_distillation_finetune.sh -o1 lr 1e-5 -o2 num_epochs 40 -o3 data_path /home/coder/code/video_sketch/data/sketch_distill/real_noanimal_lmdb -o4 data_repeat 100 --gpu_num 7 --time 99:00:00 --mem 10

python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/run_distillation_finetune.sh -o1 lr 1e-5 -o2 num_epochs 40 -o3 data_path /home/coder/code/video_sketch/data/sketch_distill/fake_noanimal_lmdb -o4 data_repeat 100 --gpu_num 7 --time 99:00:00 --mem 10

# inference finetune - distillation
# set ckpts /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr2e-6_ep200_bs1_ratio5_sketch_captions_slurm324

set ckpts /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep40_bs1_ratio5_repeat100_fake_noanimal_l_slurm142

for ckpt in $ckpts
    for epoch in (seq -f "%03g" 31 -4 31)
        
        set ckpt_folder "$ckpt/checkpoint_epoch_$epoch"

        # python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/batch_inference_causvid.sh -o1 ckpt_folder $ckpt_folder -o2 prompts_file /home/coder/code/video_sketch/data/custom_sketch0618/validation_prompts_detailed.txt -o3 seed 0 -o4 num_rollout 3 -o5 num_overlap_frames 3

        python sbatchgen_sh.py -f /home/coder/code/video_sketch/libs/CausVid/scripts/batch_inference_causvid.sh -o1 ckpt_folder $ckpt_folder -o2 prompts_file /home/coder/code/video_sketch/data/custom_sketch0618/validation_prompts_detailed.txt -o3 seed 0 -o4 num_rollout 3 -o5 num_overlap_frames 3

    end
end

python upload_checkpoint_results_to_wandb.py --project Causvid_dmdfinetune_inference /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr2e-6_ep50_bs1_ratio5_lmdb_slurm90

python upload_checkpoint_results_to_wandb.py --project Causvid_dmdfinetune_inference /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep50_bs1_ratio5_lmdb_slurm103

python upload_checkpoint_results_to_wandb.py --project Causvid_dmdfinetune_inference /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep40_bs1_ratio5_repeat100_real_noanimal_l_slurm123

# python upload_checkpoint_results_to_wandb.py --project Causvid_dmdfinetune_inference /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm21

# python upload_checkpoint_results_to_wandb.py --project Causvid_dmdfinetune_inference /home/coder/code/video_sketch/libs/CausVid/experiments/causvid_distillation_realLora32-fakeLora32_lr1e-5_ep100_bs1_ratio5_sketch_captions_slurm333