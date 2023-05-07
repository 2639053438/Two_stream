import argparse

parser = argparse.ArgumentParser(description='frame_spatical_code')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='learning rate decays in order to avoid overfitting')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Regularization: Improve the generalization ability of the model and reduce the possibility of overfitting')
parser.add_argument('--new_length', default=1, type=int,
                    help='length of sampled video frames')
parser.add_argument('--batch_size', default=25, type=int,
                    help='mini-batch size')
parser.add_argument('--workers', default=1, type=int,
                    help='Thread count')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=400, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save_freq', default=1, type=int,
                    help='Preservation frequency')
parser.add_argument('--resume', default='./checkpoints/frame_code', choices=["./checkpoints/frame_code", "./checkpoints/flow_code"],
                    help='path to latest checkpoint')
parser.add_argument('--arch', default=1, type=int,
                    help='Preservation frequency')