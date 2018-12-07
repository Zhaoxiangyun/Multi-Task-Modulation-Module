python train_mm.py --data_dir ./data/ \
--learning_rate 0.01  --model_def models.single_net_attention_res4 --embedding_size 256 \
--max_nrof_epochs 100 --lr_epoch 40 --random_trip True --batch_size 168  --image_size 150 --experiment_name mm
