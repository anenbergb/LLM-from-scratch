import argparse
import os
import sys
import shutil
from loguru import logger


def get_args() -> argparse.Namespace:
    EXPR_DIR = "/media/bryan/ssd01/expr/llm_from_scratch"
    parser = argparse.ArgumentParser(
        """
Run the LLM pre-training.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # BPE encoded dataset
    parser.add_argument(
        "--train-dataset",
        type=str,
        default=os.path.join(EXPR_DIR, "tokenization/TinyStoriesV2-GPT4-train.npy"),
        help="Path to the BPE tokenized dataset saved as a .npy file",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default=os.path.join(EXPR_DIR, "tokenization/TinyStoriesV2-GPT4-valid.npy"),
        help="Path to the BPE tokenized dataset saved as a .npy file",
    )
    parser.add_argument(
        "--tokenized-dataset-pickle",
        type=str,
        default=os.path.join(EXPR_DIR, "tokenization/bpe_10k_tinystories.pkl"),
        help="Path to a tokenized dataset .pkl file. The vocabulary can be recovered from this file.",
    )

    # output and logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(EXPR_DIR, "debug/run01"),
        help="Path to save the model",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        default=False,
        help="Overwrite the content of the output directory",
    )

    # data loading
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=128, help="Validation batch size")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")

    # optimizer (AdamW + Cosine LR decay schedule)
    parser.add_argument(
        "--max-train-iters",
        type=int,
        default=1000000,
        help="Maximum training iterations. This is the cosine cycle iters",
    )
    parser.add_argument("--lr-warmup-iters", type=int, default=10000, help="Warmup iterations")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to a specific checkpoint folder that the training should resume from. "
        "Training will resume from the iteration that was saved in the checkpoint. ",
    )
    parser.add_argument(
        "--evaluation-iters",
        type=int,
        default=100000,
        help="Frequency of evaluation in iterations",
    )
    parser.add_argument(
        "--limit-val-iters",
        type=int,
        default=0,
        help="(Debugging) Limit the number of validation iterations to accelerate evaluation.",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=2e-4,
        help="maximum learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
        help="Adam beta2",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--gradient-max-norm",
        type=float,
        default=1.0,
        help="Maximum gradient l2 norm for gradient clipping",
    )

    # Model parameters
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Dimensionality of the model",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=1344,
        help="Dimensionality of the feed-forward layer. This is roughly (8/3)*d_model while beineg a multiple of 64.",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter",
    )
    return parser.parse_args()


def delete_output_dir(output_dir):
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")


def train(args):
    if os.path.exists(args.output_dir):
        if args.overwrite_output_dir:
            logger.warning(
                f"Output directory ({args.output_dir}) already exists and overwrite_output_dir is set to True. "
                "Deleting the content of the output directory."
            )
            delete_output_dir(args.output_dir)
        else:
            logger.error(
                f"Output directory ({args.output_dir}) already exists and overwrite_output_dir is set to False. "
                "Please choose a different output directory."
            )
            sys.exit(1)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory ({args.output_dir}) created.")


if __name__ == "__main__":
    args = get_args()
    sys.exit(train(args))
