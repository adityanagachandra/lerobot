Train command : 

python -m train   --policy.path=lerobot/pi0   --dataset.repo_id=adungus/PickPlace   --batch_size=320   --steps=200000   --output_dir=outputs/train/pi0_pickplace   --policy.repo_id=adungus/smolVLA_200k \--wandb.enable=true --batch_size=24
