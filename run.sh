# affine registration
python adamflow_affine.py --data_dir='YOUR_DATA_DIR' --save_dir='YOUR_SAVE_DIR' --organ='liver' --method='AdamFlow' --n_pair=300 --device='cuda:0' --n_proj=4 --alpha=0.9 --beta=0.95 --eta=1e-2 --K=1500 --h=1.0 --eps=1e-10

# non-rigid registration
python adamflow_nonrigid.py --data_dir='YOUR_DATA_DIR' --save_dir='YOUR_SAVE_DIR' --organ='liver' --method='AdamFlow' --n_pair=300 --device='cuda:0' --w_lapl=2.0 --n_proj=4 --alpha=0.9 --beta=0.95 --eta_swd=0.5 --eta_cham=0.1 --K_swd=500 --K_cham=200 --h=1.0 --eps=1e-10
