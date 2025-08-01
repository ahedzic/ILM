cd benchmarking/cold_start

######## Starcraft ########
#### Cold-start Models ####

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_starcraft_ilm_gcn_3_iterations

# ILM(GAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name starcraft --input_size 25 --gnn_model GAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_starcraft_ilm_gat_3_iterations

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name starcraft --input_size 25 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_starcraft_ilm_sage_3_iterations

# ILM(RGCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name starcraft --input_size 25 --gnn_model RGCN --lr 0.001 --dropout 0.3 --l2 1e-7 --num_relations 3 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_starcraft_ilm_rgcn_3_iterations

# ILM(RGAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name starcraft --input_size 25 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 3 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_starcraft_ilm_rgat_3_iterations

# CSMDDI
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_csmddi.py --data_name starcraft --input_size 25 --hidden_channels 128 --num_relations 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_starcraft_csmddi

# DEAL
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_deal.py --data_name starcraft --input_size 25 --lr 0.001 --hidden_channels 128 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_starcraft_deal

# LEAP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leap.py --data_name starcraft --input_size 25 --lr 0.01 --dropout 0.5 --hidden_channels 256 --kill_cnt 3 --runs 10 --cold_perc 0.0 > overall_comparison/output_starcraft_leap

# Leroy
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leroy.py --data_name starcraft --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1 --runs 1 --cold_perc 0.0 > overall_comparison/output_starcraft_leroy


#### GNN Models ####
# GCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_starcraft_gcn_1_iterations

# GAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name starcraft --input_size 25 --gnn_model GAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_starcraft_gat_1_iterations

# SAGE
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name starcraft --input_size 25 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_starcraft_sage_1_iterations

# RGCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name starcraft --input_size 25 --gnn_model RGCN --lr 0.001 --dropout 0.3 --l2 1e-7 --num_relations 3 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_starcraft_rgcn_1_iterations

# RGAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name starcraft --input_size 25 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 3 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_starcraft_rgat_1_iterations

#### GNN+Pairwise Models ####
# NEOGNN
python main_cold_neognn.py --data_name starcraft --input_size 25 --gnn_model NeoGNN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_starcraft_neognn

# NCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ncn.py --dataset starcraft --input_size 25 --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.1 --gnndp 0.1 --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0  > overall_comparison/output_starcraft_ncn


#### Embedding Models ###
# MLP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name starcraft --input_size 25 --gnn_model mlp_model --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 1 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_starcraft_mlp_1_iterations


######## Flickr ########
#### Cold-start Models ####

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_flickr_ilm_gcn_3_iterations

# ILM(GAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model GAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_flickr_ilm_gat_3_iterations

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name flickr --input_size 500 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_flickr_ilm_sage_3_iterations

# ILM(RGCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name flickr --input_size 500 --gnn_model RGCN --lr 0.001 --dropout 0.3 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_flickr_ilm_rgcn_3_iterations

# ILM(RGAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name flickr --input_size 500 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 1 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_flickr_ilm_rgat_3_iterations

# LEAP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leap.py --data_name flickr --input_size 500 --lr 0.001 --dropout 0.3 --hidden_channels 128 --kill_cnt 3 --runs 10 --cold_perc 0.0 > overall_comparison/output_flickr_leap

# Leroy
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leroy.py --data_name flickr --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1 --runs 1 --cold_perc 0.0 > overall_comparison/output_flickr_leroy


#### GNN Models ####
# GCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name flickr --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.3 --l2 0 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_flickr_gcn_1_iterations

# GAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name flickr --input_size 500 --gnn_model GAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_flickr_gat_1_iterations

# SAGE
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name flickr --input_size 500 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_flickr_sage_1_iterations

# RGCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name flickr --input_size 500 --gnn_model RGCN --lr 0.001 --dropout 0.3 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_flickr_rgcn_1_iterations

# RGAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name flickr --input_size 500 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 1 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_flickr_rgat_1_iterations

#### GNN+Pairwise Models ####
# NEOGNN
python main_cold_neognn.py --data_name flickr --input_size 500 --gnn_model NeoGNN --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_flickr_neognn

# NCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ncn.py --dataset flickr --input_size 500 --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.3 --gnndp 0.3 --mplayers 2 --nnlayers 3 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0  > overall_comparison/output_flickr_ncn


#### Embedding Models ###
# MLP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name flickr --input_size 500 --gnn_model mlp_model --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 1 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_flickr_mlp_1_iterations


######## Yelp ########
#### Cold-start Models ####

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name yelp --input_size 300 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_yelp_ilm_gcn_3_iterations

# ILM(GAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name yelp --input_size 300 --gnn_model GAT --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_yelp_ilm_gat_3_iterations

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name yelp --input_size 300 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_yelp_ilm_sage_3_iterations

# ILM(RGCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_yelp_ilm_rgcn_3_iterations

# ILM(RGAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name yelp --input_size 300 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_yelp_ilm_rgat_3_iterations

# Leroy
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leroy.py --data_name yelp --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1 --runs 1 --cold_perc 0.0 > overall_comparison/output_yelp_leroy


#### GNN Models ####
# GCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name yelp --input_size 300 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_yelp_gcn_1_iterations

# GAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name yelp --input_size 300 --gnn_model GAT --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_yelp_gat_1_iterations

# SAGE
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name yelp --input_size 300 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_yelp_sage_1_iterations

# RGCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name yelp --input_size 300 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_yelp_rgcn_1_iterations

# RGAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name yelp --input_size 300 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_yelp_rgat_1_iterations


#### Embedding Models ###
# MLP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name yelp --input_size 300 --gnn_model mlp_model --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_yelp_mlp_1_iterations


######## CSMDDI ########
#### Cold-start Models ####

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_csmddi_ilm_gcn_3_iterations

# ILM(GAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model GAT --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 3 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_csmddi_ilm_gat_3_iterations

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_csmddi_ilm_sage_3_iterations

# ILM(RGCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name csmddi --input_size 1493 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 87 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_csmddi_ilm_rgcn_3_iterations

# ILM(RGAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name csmddi --input_size 1493 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 87 --num_layers 3 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_csmddi_ilm_rgat_3_iterations

# CSMDDI
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_csmddi.py --data_name csmddi --input_size 1493 --hidden_channels 128 --num_relations 87 --epochs 9999 --kill_cnt 3 --eval_steps 1 --runs 10 --cold_perc 0.0 > overall_comparison/output_csmddi_csmddi

# DEAL
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_deal.py --data_name csmddi --input_size 1493 --lr 0.01 --hidden_channels 256 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_csmddi_deal

# LEAP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leap.py --data_name csmddi --input_size 1493 --lr 0.001 --dropout 0.5 --hidden_channels 128 --kill_cnt 3 --runs 10 --cold_perc 0.0 > overall_comparison/output_csmddi_leap

# Leroy
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leroy.py --data_name csmddi --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1 --runs 1 --cold_perc 0.0 > overall_comparison/output_csmddi_leroy


#### GNN Models ####
# GCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name csmddi --input_size 1493 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_csmddi_gcn_1_iterations

# GAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name csmddi --input_size 1493 --gnn_model GAT --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 3 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_csmddi_gat_1_iterations

# SAGE
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name csmddi --input_size 1493 --gnn_model SAGE --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_csmddi_sage_1_iterations

# RGCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name csmddi --input_size 1493 --gnn_model RGCN --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 87 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_csmddi_rgcn_1_iterations

# RGAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name csmddi --input_size 1493 --gnn_model RGAT --lr 0.001 --dropout 0.1 --l2 1e-4 --num_relations 87 --num_layers 3 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_csmddi_rgat_1_iterations

#### GNN+Pairwise Models ####
# NEOGNN
python main_cold_neognn.py --data_name csmddi --input_size 1493 --gnn_model NeoGNN --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 1 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_csmddi_neognn

# NCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ncn.py --dataset csmddi --input_size 1493 --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.3 --gnndp 0.3 --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0  > overall_comparison/output_csmddi_ncn


#### Embedding Models ###
# MLP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name csmddi --input_size 1493 --gnn_model mlp_model --lr 0.01 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_csmddi_mlp_1_iterations


######## Pubmed ########
#### Cold-start Models ####

# ILM(GCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_pubmed_ilm_gcn_3_iterations

# ILM(GAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model GAT --lr 0.001 --dropout 0.3 --l2 0 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_pubmed_ilm_gat_3_iterations

# ILM(SAGE)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name pubmed --input_size 500 --gnn_model SAGE --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_pubmed_ilm_sage_3_iterations

# ILM(RGCN)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name pubmed --input_size 500 --gnn_model RGCN --lr 0.001 --dropout 0.3 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_pubmed_ilm_rgcn_3_iterations

# ILM(RGAT)
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_ilm.py --data_name pubmed --input_size 500 --gnn_model RGAT --lr 0.001 --dropout 0.3 --l2 0 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0 > overall_comparison/output_pubmed_ilm_rgat_3_iterations

# CSMDDI
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_csmddi.py --data_name pubmed --input_size 500 --hidden_channels 128 --num_relations 1 --epochs 9999 --kill_cnt 3 --eval_steps 1 --runs 10 --cold_perc 0.0 > overall_comparison/output_pubmed_csmddi

# DEAL
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_deal.py --data_name pubmed --input_size 500 --lr 0.01 --hidden_channels 256 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_pubmed_deal

# LEAP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leap.py --data_name pubmed --input_size 500 --lr 0.001 --dropout 0.5 --hidden_channels 128 --kill_cnt 3 --runs 10 --cold_perc 0.0 > overall_comparison/output_pubmed_leap

# Leroy
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_leroy.py --data_name pubmed --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1 --runs 1 --cold_perc 0.0 > overall_comparison/output_pubmed_leroy


#### GNN Models ####
# GCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name pubmed --input_size 500 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_pubmed_gcn_1_iterations

# GAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name pubmed --input_size 500 --gnn_model GAT --lr 0.001 --dropout 0.3 --l2 0 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_pubmed_gat_1_iterations

# SAGE
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name pubmed --input_size 500 --gnn_model SAGE --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_pubmed_sage_1_iterations

# RGCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name pubmed --input_size 500 --gnn_model RGCN --lr 0.001 --dropout 0.3 --l2 1e-7 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_pubmed_rgcn_1_iterations

# RGAT
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_relational_gnn.py --data_name pubmed --input_size 500 --gnn_model RGAT --lr 0.001 --dropout 0.3 --l2 0 --num_relations 1 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_pubmed_rgat_1_iterations

#### GNN+Pairwise Models ####
# NEOGNN
python main_cold_neognn.py --data_name pubmed --input_size 500 --gnn_model NeoGNN --lr 0.01 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --cold_perc 0.0 > overall_comparison/output_pubmed_neognn

# NCN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ncn.py --dataset pubmed --input_size 500 --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.3 --gnndp 0.3 --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0  > overall_comparison/output_pubmed_ncn

#### Embedding Models ###
# MLP
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_gnn.py --data_name pubmed --input_size 500 --gnn_model mlp_model --lr 0.01 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 1 --cold_perc 0.0 > overall_comparison/output_pubmed_mlp_1_iterations
