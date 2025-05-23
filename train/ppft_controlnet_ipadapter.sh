export id=0
export CUDA_VISIBLE_DEVICES=$id
accelerate launch --mixed_precision="fp16" ppft_controlnet_ipadapter.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_path="output_train_controlnet_ipadapter" \
 --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
 --ip_ckpt="output_train_controlnet_ipadapter/ip_adapter.bin" \
 --train_data_dir="Gustavosta-sample" \
 --proportion_empty_prompts=1 \
 --validation_prompt "" \
 --validation_image "COCO2017test/000000000071.jpg" "COCO2017test/000000000074.jpg" \
 --num_validation_images=5 \
 --validation_steps=500 \
 --output_dir="output_ppft_controlnet_ipadapter" \
 --seed=2048 \
 --num_train_epochs=30 \
 --train_batch_size=20 \
 --checkpointing_steps=500 \
 --start_from_pretrain="../AquaLoRA-Models/pretrained_latent_watermark/pretained_latentwm.pth" \
 --learning_rate=1e-4 \
 --lr_scheduler="cosine_with_restarts" \
 --lr_warmup_steps=0 \
 --lr_end=0.01 \
 --report_to="wandb" \
 --rank=320 \
 --msg_bits=48 \
 >ppft_controlnet_ipadapter_$id.log 2>&1