import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training for Knowledge Distillation')

    parser.add_argument('--current_time', default='', type=str)
    # Datasets
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=42, type=int,help='seed to use')

    parser.add_argument('--T', default=20, type=float, help='temperature to use for KD')
    parser.add_argument('--Lambda', default=0.9, type=float, help='distilling weight to use for KD')

    # Optimization options
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--exp_name', default="", type=str,
                        help='Experiment name')

    parser.add_argument('--teacher', default="", type=str, help='Experiment name')
    parser.add_argument('--teacher_path', default="", type=str, help='Experiment name')

    parser.add_argument('--train-batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch-size', default=100, type=int, metavar='N',
                        help='test batchsize')

    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--alpha', default=5.0, type=float, help='alpha to train Label Smoothing with')
    parser.add_argument('--beta', default=1.0, type=float, help='beta to train MDCA with')
    parser.add_argument('--gamma', default=1, type=float, help='gamma to train Focal Loss with')

    parser.add_argument('--scheduler', default="multistep", type=str, help='scheduler to use for training')
    parser.add_argument('--schedule-steps', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, help='LR is multiplied by this on schedule.')
    parser.add_argument('--warmup', default=0, type=int, 
        help='warmup to use for training with scheduler. Should be less than 0.1 of total training time'
    )

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    # Checkpoints
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--loss', default='cross_entropy', type=str)
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('--optimizer', default='sgd', type=str)

    parser.add_argument('--prefix', default='', type=str, metavar='PRNAME')
    parser.add_argument('--regularizer', default='l2', type=str, metavar='RNAME')
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--chain_length', default=None, type=int)
    
    return parser.parse_args()