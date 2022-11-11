import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description = '''A Simple Diffusion Model''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed',
                        type = int, default = 0,
                        help = 'Random seed, < 0 is no seeding')

    parser.add_argument('--n_epochs',
                        type = int, default = 10000,
                        help = 'How many epochs')
    
    parser.add_argument('--n_samples',
                        type = int, default = 10000,
                        help = 'Num of generated samples')

    parser.add_argument('--batch_size',
                        type = int, default = 2048,
                        help = 'Batch size')

    parser.add_argument('--learning_rate',
                        type = float, default = 1e-4,
                        help = 'Learning rate')

    parser.add_argument('--T',
                        type = int, default = 1000,
                        help = 'Timestamp')
    
    parser.add_argument('--beta0',
                        type = float, default = 0.0001,
                        help = 'Initial beta value')
    
    parser.add_argument('--betaT',
                        type = float, default = 0.02,
                        help = 'Final beta value')
    
    return parser.parse_args()


def scale_data(data, xmin, xmax, rmin, rmax):
    return ((rmax - rmin) * (data - xmin) / (xmax - xmin)) + rmin

def unscale_data(data, xmin, xmax, rmin, rmax):
    return ((data - rmin) * (xmax - xmin) / (rmax - rmin)) + xmin