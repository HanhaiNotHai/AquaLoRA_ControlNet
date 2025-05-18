accelerate launch --mixed_precision="fp16" train_controlnet_aqua.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="output4" \
 --train_batch_size=16 \
 --num_train_epochs=25 \
 --checkpointing_steps=500 \
 --resume_from_lora="../AquaLoRA-Models/ppft_trained" \
 --start_from_pretrain="../AquaLoRA-Models/pretrained_latent_watermark/pretained_latentwm.pth" \
 --learning_rate=1e-5 \
 --lr_scheduler="constant" \
 --report_to="wandb" \
 --train_data_dir="Gustavosta-sample" \
 --proportion_empty_prompts=1 \
 --validation_prompt "" \
 --validation_image "COCO2017test/000000000071.jpg" "COCO2017test/000000000074.jpg" \
 --num_validation_images=4 \
 --validation_steps=500 \
 --rank=320 \
 --msg_bits=48 \
 >train_controlnet_aqua.log 2>&1