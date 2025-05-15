accelerate launch --mixed_precision="fp16" train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
 --output_dir="output_train_controlnet" \
 --train_batch_size=16 \
 --num_train_epochs=20 \
 --checkpointing_steps=500 \
 --learning_rate=1e-4 \
 --lr_scheduler="constant" \
 --report_to="wandb" \
 --train_data_dir="Gustavosta-sample" \
 --proportion_empty_prompts=1 \
 --validation_prompt "" \
 --validation_image "COCO2017test/000000000071.jpg" "COCO2017test/000000000074.jpg" \
 --num_validation_images=4 \
 --validation_steps=500 \
 >train_controlnet.log 2>&1