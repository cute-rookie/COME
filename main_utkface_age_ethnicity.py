import argparse
import logging
import os, time
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import configs
from scripts import train_hashing

def main(file_name=None):
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S')

    torch.backends.cudnn.benchmark = True
    configs.default_workers = os.cpu_count()

    parser = argparse.ArgumentParser(description='OrthoHash')
    parser.add_argument('--nbit', default=256, type=int, help='number of bits')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--arch', default='alexnet', choices=['alexnet'], help='backbone name')

    ############## dataset related ##################
    parser.add_argument('--data_root', default='./data/UTKFace/', type=str, help='dataset root') #
    parser.add_argument('--generated_image_root', default='./dataset/generated_data/', type=str, help='dataset root')
    parser.add_argument('--pseudo_image_root', default='./dataset/pseudo_data', type=str, help='dataset root')
    parser.add_argument('--ds', default='utkface', choices=['utkface', 'celeba'], help='dataset')
    parser.add_argument('--nclass', default=5, type=int, help='number of classes')
    parser.add_argument('--class_type', default=0, type=int, help='category of multi-class classification')
    parser.add_argument('--ta', default='age', choices=['age', 'gender', 'ethnicity'], help='target attribute')
    parser.add_argument('--sa', default='ethnicity', choices=['age', 'ethnicity'], help='sensitive attribute')
    parser.add_argument('--labeled_sample_num', default=600, type=int, help='number of labeled samples')
    parser.add_argument('--generate_labeled_sample', default=False, type=bool, help='generate labeled samples using conditional diffusion model, with same target attribute and different sensitive attribute')
    parser.add_argument('--image_local_find', default=False, type=bool, help='find local images for each labeled sample')
    parser.add_argument('--LLM_for_image_type', default=0, help='choose the type of LLM to generate samples')
    parser.add_argument('--reset', default=False, type=bool, help='reset the dataset')

    ############## augmentation ##################
    parser.add_argument('--aug_for_labeled_sample', default=True, type=bool, help='use augmenation image of labeled data samples')

    ############## annotation unlabeled data ###########
    parser.add_argument('--annotation_unlabeled_data', default=True, type=bool, help='annotation image of unlabeled data samples')
    parser.add_argument('--low_confidence_threshold', default=0.2, type=float, help='the confidence threshold for low confidence samples in CLIP')
    parser.add_argument('--high_confidence_threshold', default=0.8, type=float, help='the confidence threshold for high confidence samples in LLM annotation')
    parser.add_argument('--labeled_LLM', default='Qwen_2_5_72B', choices=['Qwen_2_5_7B', 'Qwen_2_5_72B'], type=str, help='the name of LLM model to annotate the low-confidence samples')

    # transformer related
    parser.add_argument('--hidden-dim', default=256, type=int, help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--enc_layers', default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dropout', default=0.0, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--enc_n_points', default=4, type=int, help="number of deformable attention sampling points in encoder layers")
    parser.add_argument('--use_grl', default=False, type=bool, help="whether to use gradient reversal layer in transformer")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")

    # loss related
    parser.add_argument('--scale', default=8, type=float, help='scale for cos-sim')
    parser.add_argument('--margin', default=0.3, type=float, help='ortho margin')
    parser.add_argument('--margin-type', default='cos', choices=['cos', 'arc'], help='margin type')
    parser.add_argument('--ce', default=1.0, type=float, help='classification scale')
    parser.add_argument('--final_feature', default=1.0, type=float, help='loss hyperparameter for MI final feature')
    parser.add_argument('--lvl', default=0.01, type=float, help='loss hyperparameter for level-wise classification')
    parser.add_argument('--attributes_feature', default=0.1, type=float, help='loss hyperparameter for MI attributes feature')
    parser.add_argument('--multiclass-loss', default='label_smoothing', choices=['bce', 'imbalance', 'label_smoothing'], help='multiclass loss types')

    # codebook generation
    parser.add_argument('--codebook-method', default='N', choices=['N', 'B', 'O'], help='N = sign of gaussian; '
                                                                                        'B = bernoulli; '
                                                                                        'O = optimize')

    parser.add_argument('--seed', default=42, help='seed number; default: random')

    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    # judge the number of classes, target attribute, sensitive attribute
    args.class_type = configs.reasonableness_judgment(args.ds, args.nclass, args.ta, args.sa)
    args.generated_image_root = os.path.join(args.generated_image_root, f'{args.ds}_{args.ta}_{args.sa}_{args.nclass}')
    args.pseudo_image_root = os.path.join(args.pseudo_image_root, f'{args.ds}_{args.ta}_{args.sa}_{args.nclass}')

    config = {
        'file_name': file_name,
        'arch': args.arch,
        'aug_for_labeled_sample': args.aug_for_labeled_sample,
        'annotation_unlabeled_data': args.annotation_unlabeled_data,
        'low_confidence_threshold': args.low_confidence_threshold,
        'high_confidence_threshold': args.high_confidence_threshold,
        'labeled_LLM': args.labeled_LLM,
        'class_type': args.class_type,
        'arch_kwargs': {
            'nbit': args.nbit,
            'nclass': args.nclass,  # will be updated below
            'pretrained': True,
            'freeze_weight': False,
            'hidden_dim': args.hidden_dim,
            'nheads': args.nheads,
            'enc_layers': args.enc_layers,
            'dropout': args.dropout,
            'enc_n_points': args.enc_n_points,
            'use_grl': args.use_grl,
            'position_embedding': args.position_embedding
        },
        'sample': {
            'labeled_sample_num': args.labeled_sample_num,
            'generate_labeled_sample': args.generate_labeled_sample,
            'LLM_for_image_type': args.LLM_for_image_type,
        },
        'image_local_find': args.image_local_find,
        'batch_size': args.bs,
        'dataset': args.ds,
        'multiclass': args.ds == 'nuswide',
        'dataset_kwargs': {
            'resize': 224,
            'crop': 224,
            'norm': 2,
            'evaluation_protocol': 1,  # only affect cifar10
            'reset': args.reset,
            'separate_multiclass': False,
            'data_root': args.data_root,
            'generated_image_root': args.generated_image_root,
            'pseudo_image_root': args.pseudo_image_root,
            'target_attribute': args.ta,
            'sensitive_attribute': args.sa,
        },
        'optim': 'adam',
        'optim_kwargs': {
            'lr': args.lr,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'nesterov': False,
            'betas': (0.9, 0.999)
        },
        'epochs': args.epochs,
        'scheduler': 'mstep',
        'scheduler_kwargs': {
            'step_size': int(args.epochs * 0.8),
            'gamma': 0.1,
            'milestones': '0.25,0.5,0.75'
        },
        'save_interval': 0,
        'eval_interval': 1,
        'seed': args.seed,

        'codebook_generation': args.codebook_method,

        # loss_param
        'loss_params':{
            'ce': args.ce,
            'final_feature': args.final_feature,
            'lvl': args.lvl,
            'attributes_feature': args.attributes_feature,
        },
        's': args.scale,
        'm': args.margin,
        'm_type': args.margin_type,
        'multiclass_loss': args.multiclass_loss,
        'device': args.device,
        'multiclass_loss': args.multiclass_loss,
    }

    config['R'] = configs.R(config)

    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    logdir = (f'logs/{time_str}_compare_diff_loss_'
              f'dim_{config["arch_kwargs"]["hidden_dim"]}_seed_{config["seed"]}_'
              f'R_{config["R"]}_'
              f'labeled_sample_{config["sample"]["labeled_sample_num"]}_'
              f'generate_image_{config["aug_for_labeled_sample"]}_'
              f'annotation_unlabeled_data_{config["annotation_unlabeled_data"]}_'
              f'low_confidence_threshold_{config["low_confidence_threshold"]}_'
              f'high_confidence_threshold_{config["high_confidence_threshold"]}_'
              f'{config["dataset"]}_ta_{config["dataset_kwargs"]["target_attribute"]}_'
              f'sa_{config["dataset_kwargs"]["sensitive_attribute"]}_'
              f'nclass_{config["arch_kwargs"]["nclass"]}_'
              f'epoch_{config["epochs"]}')


    config['logdir'] = logdir
    train_hashing.main(config)

if __name__ == '__main__':
    # Take the file name as input
    main()

