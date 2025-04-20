# Cityscapes to Foggy Cityscapes
GPUS_PER_NODE=2 MASTER_PORT=21001 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_foggy_r50.yaml \
--opts DATASET.AOOD_SETTING 1 OUTPUT_DIR experiments/city_to_foggy/setting1

GPUS_PER_NODE=2 MASTER_PORT=21001 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_foggy_r50.yaml \
--opts DATASET.AOOD_SETTING 2 OUTPUT_DIR experiments/city_to_foggy/setting2

GPUS_PER_NODE=2 MASTER_PORT=21001 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_foggy_r50.yaml \
--opts DATASET.AOOD_SETTING 3 OUTPUT_DIR experiments/city_to_foggy/setting3

CUDA_VISIBLE_DEVICES=2,3 GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_foggy_r50.yaml --opts DATASET.AOOD_SETTING 4 OUTPUT_DIR experiments/city_to_foggy/setting4-memcos
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 MASTER_PORT=21002 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_foggy_r50.yaml --opts DATASET.AOOD_SETTING 1 OUTPUT_DIR experiments/city_to_foggy/setting1-memcos+ext

# Pascal to CLipart

CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2  MASTER_PORT=21002 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_pascal_to_clipart_r50.yaml --opts OUTPUT_DIR experiments/pascal_to_clipart/cos+extG
# Cityscapes to BDD00k_daytime


CUDA_VISIBLE_DEVICES=2,3 GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_bdd100k_r50.yaml --opts DATASET.AOOD_SETTING 1 OUTPUT_DIR experiments/city_to_bdd100k/setting1-cos

GPUS_PER_NODE=2 MASTER_PORT=21001 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_bdd100k_r50.yaml \
--opts DATASET.AOOD_SETTING 2 OUTPUT_DIR experiments/city_to_bdd100k/setting2

GPUS_PER_NODE=2 MASTER_PORT=21001 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_bdd100k_r50.yaml \
--opts DATASET.AOOD_SETTING 3 OUTPUT_DIR experiments/city_to_bdd100k/setting3

GPUS_PER_NODE=2 MASTER_PORT=21001 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_bdd100k_r50.yaml \
--opts DATASET.AOOD_SETTING 4 OUTPUT_DIR experiments/city_to_bdd100k/setting4