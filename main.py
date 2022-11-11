import torch
import numpy as np
from display import plot_2D_scatter

from sample_generator import generate_spiral_data
from utils import parse_args
from trainer import Trainer


if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("#" * 20)
    print(f"Device: {device}")
    print("#" * 20)

    # generate training samples
    X_train = generate_spiral_data(args.n_samples).to(device)
    

    trainer = Trainer(device, args.n_epochs, args.batch_size, args.learning_rate, args.T,
                      args.beta0, args.betaT)

    trainer.train(X_train)
    
    X_test = torch.randn((1000, 2), device=device)
    X_test_pred = trainer.test(X_test)

    plot_2D_scatter(X_train, X_test_pred)

    

