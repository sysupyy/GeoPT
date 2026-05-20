import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

parser = argparse.ArgumentParser('Fine-Tuning Neural Simulators')

## training
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='maximum epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
parser.add_argument('--pct_start', type=float, default=0.3, help='oncycle lr schedule')
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None, help='make the training stable')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type, select from Adam, AdamW')
parser.add_argument('--scheduler', type=str, default='OneCycleLR',
                    help='learning rate scheduler, select from [OneCycleLR, CosineAnnealingLR, StepLR]')
parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay parameter for StepLR scheduler')

## data
parser.add_argument('--data_path', type=str, default='./data', help='data folder, should change accordingly')
parser.add_argument('--loader', type=str, default='airfoil', help='type of data loader')
parser.add_argument('--ntrain', type=int, default=1000, help='training data numbers')
parser.add_argument('--ntest', type=int, default=200, help='test data numbers')
parser.add_argument('--normalize', type=bool, default=False, help='make normalization to output')
parser.add_argument('--norm_type', type=str, default='UnitTransformer',
                    help='dataset normalize type. select from [UnitTransformer, UnitGaussianNormalizer]')
parser.add_argument('--geotype', type=str, default='unstructured',
                    help='select from [unstructured, structured_1D, structured_2D, structured_3D]')
parser.add_argument('--space_dim', type=int, default=2, help='position information dimension')
parser.add_argument('--fun_dim', type=int, default=0, help='input observation dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output observation dimension')

## task
parser.add_argument('--task', type=str, default='steady',
                    help='select from [steady, GeoPT_finetune]')
parser.add_argument('--dynamics', type=str, default='hull',
                    help='select from [hull, craft, drivAerml, nasa, crash]')
## models
parser.add_argument('--model', type=str, default='Transolver')
parser.add_argument('--n_hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n_layers', type=int, default=3, help='layers')
parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--mlp_ratio', type=int, default=1, help='mlp ratio for feedforward layers')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--checkpoint', type=int, default=0, help='using gradient checkpoint or not')

## model specific configuration
parser.add_argument('--slice_num', type=int, default=32, help='number of physical states for Transolver')
parser.add_argument('--use_geo_film', type=int, default=0,
                    help='use geometry-conditioned FiLM on Transolver residual branches')
parser.add_argument('--geo_film_hidden', type=int, default=128, help='hidden dim for geometry context encoder')
parser.add_argument('--geo_film_strength', type=float, default=0.1,
                    help='initial scale for geometry FiLM residual modulation')
parser.add_argument('--use_local_adaptive_slice', type=int, default=0,
                    help='use geometry-conditioned local adaptive slice bias in Physics-Attention')
parser.add_argument('--local_slice_strength', type=float, default=1.0,
                    help='initial scale for adaptive slice bias')
parser.add_argument('--physics_mixer', type=str, default='transolver',
                    choices=['transolver', 'linearno', 'gated_linear'],
                    help='physical-state token mixer: original Transolver attention or LinearNO-style variants')
parser.add_argument('--use_eidetic_slice', type=int, default=0,
                    help='use Transolver++ style adaptive temperature for eidetic physical states')
parser.add_argument('--eidetic_min_temp', type=float, default=0.01,
                    help='minimum temperature for eidetic slice assignment')
parser.add_argument('--eidetic_gumbel', type=int, default=0,
                    help='use Gumbel-softmax slice assignment during training')
parser.add_argument('--eidetic_hard', type=int, default=0,
                    help='use straight-through hard Gumbel slice assignment')
parser.add_argument('--use_condition_adapter', type=int, default=0,
                    help='enable zero-init downstream condition adapters inside Transolver hidden blocks')
parser.add_argument('--adapter_layers', type=str, default='1,3,5',
                    help='comma-separated hidden block indices where condition adapters are injected')
parser.add_argument('--adapter_hidden_dim', type=int, default=128,
                    help='hidden width for downstream condition adapters')
parser.add_argument('--adapter_condition_source', type=str, default='x_fx',
                    choices=['x', 'fx', 'x_fx'],
                    help='condition features used by adapters')
parser.add_argument('--lambda_adapter', type=float, default=0.0,
                    help='optional L2 regularization weight for adapter residuals')
parser.add_argument('--use_output_residual_head', type=int, default=0,
                    help='enable zero-init output residual head conditioned on hidden states and input features')
parser.add_argument('--output_residual_hidden', type=int, default=256,
                    help='hidden width for the output residual head')
parser.add_argument('--use_prompt_nonlocal', type=int, default=0,
                    help='enable zero-init non-local token mixer inside the prompt-side dynamics MLP')
parser.add_argument('--prompt_nonlocal_context', type=int, default=128,
                    help='number of context tokens sampled by prompt non-local mixer')
parser.add_argument('--prompt_nonlocal_reduction', type=int, default=2,
                    help='channel reduction ratio in prompt non-local mixer')

## eval
parser.add_argument('--eval', type=int, default=0, help='evaluation or not')
parser.add_argument('--save_name', type=str, default='Transolver_check', help='name of folders')
parser.add_argument('--vis_num', type=int, default=10, help='number of visualization cases')
parser.add_argument('--vis_bound', type=int, nargs='+', default=None, help='size of region for visualization, in list')

## finetune
parser.add_argument('--finetune', type=int, default=0, help='finetune or not')
parser.add_argument('--finetune_name', type=str, default='Transolver_check', help='name of folders')

args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():

    if args.task == 'GeoPT_finetune':
        from exp.GeoPT_finetune import Exp_Steady
        exp = Exp_Steady(args)
    elif args.task == 'steady_cond':
        from exp.steady_cond import Exp_Steady
        exp = Exp_Steady(args)
    else:
        raise ValueError('task not supported')

    if eval:
        exp.test()
        exp.test_full_mesh()
    else:
        exp.train()
        exp.test()
        exp.test_full_mesh()

if __name__ == "__main__":
    main()
