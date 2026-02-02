cd benchmarking/cold_start

######## Starcraft ########

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm_faster.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --weights meta --cold_perc 0.0 > iterations/output_starcraft_ilm_gcn_1_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm_faster.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 2 --weights meta --cold_perc 0.0 > iterations/output_starcraft_ilm_gcn_2_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm_faster.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > iterations/output_starcraft_ilm_gcn_3_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm_faster.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 4 --weights meta --cold_perc 0.0 > iterations/output_starcraft_ilm_gcn_4_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm_faster.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 5 --weights meta --cold_perc 0.0 > iterations/output_starcraft_ilm_gcn_5_iterations


######## Flickr ########

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --weights meta --cold_perc 0.0 > iterations/output_flickr_ilm_gcn_1_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 2 --weights meta --cold_perc 0.0 > iterations/output_flickr_ilm_gcn_2_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > iterations/output_flickr_ilm_gcn_3_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 4 --weights meta --cold_perc 0.0 > iterations/output_flickr_ilm_gcn_4_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 5 --weights meta --cold_perc 0.0 > iterations/output_flickr_ilm_gcn_5_iterations

######## Yelp ########

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --weights meta --cold_perc 0.0 > iterations/output_yelp_ilm_rgcn_1_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 2 --weights meta --cold_perc 0.0 > iterations/output_yelp_ilm_rgcn_2_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > iterations/output_yelp_ilm_rgcn_3_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 4 --weights meta --cold_perc 0.0 > iterations/output_yelp_ilm_rgcn_4_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 5 --weights meta --cold_perc 0.0 > iterations/output_yelp_ilm_rgcn_5_iterations


######## CSMDDI ########

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --weights meta --cold_perc 0.0 > iterations/output_csmddi_ilm_sage_1_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 2 --weights meta --cold_perc 0.0 > iterations/output_csmddi_ilm_sage_2_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > iterations/output_csmddi_ilm_sage_3_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 4 --weights meta --cold_perc 0.0 > iterations/output_csmddi_ilm_sage_4_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 5 --weights meta --cold_perc 0.0 > iterations/output_csmddi_ilm_sage_5_iterations


######## Pubmed ########

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --weights meta --cold_perc 0.0 > iterations/output_pubmed_ilm_gcn_1_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 2 --weights meta --cold_perc 0.0 > iterations/output_pubmed_ilm_gcn_2_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > iterations/output_pubmed_ilm_gcn_3_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 4 --weights meta --cold_perc 0.0 > iterations/output_pubmed_ilm_gcn_4_iterations
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 5 --weights meta --cold_perc 0.0 > iterations/output_pubmed_ilm_gcn_5_iterations