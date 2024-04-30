import argparse

def parse_arguments():
    """Argument parser for batch submission during model testing"""
    # Create an argument parser
    parser = argparse.ArgumentParser(description='AHA Segmentation Training')

    # Add command-line arguments
    parser.add_argument('--data_path', type=str, default='./data/standard_labels/train', help='Path to the data')
    parser.add_argument('--val_data_path', type=str, default='./data/standard_labels/val', help='Path to the validation data') #CHANGE LATER
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_path', type=str, default='./model_tests', help='Path to the results folder')
    parser.add_argument('--no_midpoint', action="store_true", help='Toggling whether to include midsegment data')
    parser.add_argument('--test_name_prefix', type=str, default="", help='Something to add onto before the test name')
    parser.add_argument('--filter_level', type=int, default=0, help='0 = no filter, 1 = filter terrible, 2 = filter questionable and terrible')
    parser.add_argument('--record_spread', action="store_true", help='Records spread')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=600, help='Number of epochs')
    parser.add_argument('--num_features', type=int, default=8, help='Number features in CNN')
    parser.add_argument('--relu', action="store_true", help='Toggle relu')
    parser.add_argument('--dropout', type=float, default=0.5, help='Vary dropout amount')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='When to early stop')

    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for UNet')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay for UNet')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum for UNet')
    parser.add_argument('--bilinear', action="store_true", help='Bilinear?')
    parser.add_argument('--sigma', type=float, default=1.5, help='Sigma for UNet')
    parser.add_argument('--CE', type=float, default=0.8, help='Cross Entropy weighting for UNet')
    parser.add_argument('--tversky_beta', type=float, default=-1, help='Tversky beta (F_beta score)')
    parser.add_argument('--amp', action="store_true", help='amp variable for unet')
    parser.add_argument('--save_checkpoint', action="store_true", help='Save UNet checkpoint?')

    parser.add_argument('--rotation', type=int, default=0, help='Rotation amount')
    parser.add_argument('--translation', type=float, default=0.0, help='Translation amount')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale amount')
    parser.add_argument('--contrast', type=float, default=1.0, help='Constrast amount')
    parser.add_argument('--flipping', action="store_true", help='Flip?')


    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments using args.data_path, args.random_seed, args.batch_size, etc.
    data_path = args.data_path
    val_data_path = args.val_data_path
    random_seed = args.random_seed
    results_path = args.results_path
    no_midpoint = args.no_midpoint
    test_name_prefix = args.test_name_prefix
    filter_level = args.filter_level
    record_spread = args.record_spread

    # Model variables
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_features = args.num_features
    relu = args.relu
    dropout = args.dropout
    early_stopping_patience = args.early_stopping_patience

    # UNet variables
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    bilinear = args.bilinear
    sigma = args.sigma
    CE = args.CE
    tversky_beta = args.tversky_beta
    amp = args.amp
    save_checkpoint = args.save_checkpoint

    # Data augmentation variables
    rotation = args.rotation
    translation = args.translation
    scale = args.scale
    contrast = args.contrast
    flipping = args.flipping


    return data_path, val_data_path, random_seed, results_path, no_midpoint, test_name_prefix, filter_level, record_spread, \
        batch_size, num_epochs, num_features, relu, dropout, early_stopping_patience, \
            lr, weight_decay, momentum, bilinear, sigma, CE, tversky_beta, amp, save_checkpoint, \
                rotation, translation, scale, contrast, flipping