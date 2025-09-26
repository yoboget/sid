import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='qm9H',
        help="Name of the dataset. Available:  qm9, zinc, planar, sbm"
    )

    parser.add_argument(
        "--work_type", type=str,
        default='train', help="Options: train or sample"
    )

    parser.add_argument(
        "--train_model", type=bool, default=True, help="Whether to train model"
    )

    parser.add_argument(
        "--train_critic", type=bool, default=False, help="Whether to train the critic"
    )

    parser.add_argument(
        "--wandb", type=str,
        default='no', help="If W&B is online, offline or disabled"
    )

    parser.add_argument(
        "--denoiser_dir", type=str,
        default=None, help="Path to the model directory"
    )

    parser.add_argument(
        "--critic_dir", type=str,
        default=None, help="Path to the model directory"
    )

    return parser.parse_args()
