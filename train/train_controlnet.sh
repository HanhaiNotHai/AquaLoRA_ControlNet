accelerate launch --mixed_precision="fp16" train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="output_train_controlnet" \
 --train_batch_size=32 \
 --num_train_epochs=30 \
 --checkpointing_steps=500 \
 --learning_rate=1e-5 \
 --lr_scheduler="cosine_with_restarts" \
 --lr_warmup_steps=0 \
 --report_to="wandb" \
 --train_data_dir="Gustavosta-sample" \
 --proportion_empty_prompts=1 \
 --validation_prompt "" \
 --validation_image "COCO2017test/000000000071.jpg" "COCO2017test/000000000074.jpg" \
 --num_validation_images=4 \
 --validation_steps=500 \
 >train_controlnet.log 2>&1