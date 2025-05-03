import argparse
import os
import sys
import shutil
from loguru import logger
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast

from fvcore.nn import parameter_count_table, FlopCountAnalysis

from llm.tokenization import Tokenizer
from llm.transformer import TransformerLM
from llm.data import random_training_iterator, SequentialValidationDataset
from llm.optimizer import get_lr_cosine_schedule, AdamW
from llm.nn_utils import cross_entropy, perplexity, gradient_clipping
from llm.serialization import load_checkpoint, save_checkpoint
from llm.generation import generateLLM

VAL_PROMPTS = [
    "Once upon a time there was a little boy named Ben. Ben loved to",
]


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
    parser.add_argument("--special-tokens", type=str, nargs="+", default=["<|endoftext|>"])

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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to a specific checkpoint folder that the training should resume from. "
        "Training will resume from the iteration that was saved in the checkpoint. ",
    )
    parser.add_argument(
        "--checkpoint-iters",
        type=int,
        default=10000,
        help="Checkpoint every N iterations",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Precision to use for training.",
    )

    # data loading
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
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
    parser.add_argument(
        "--weight-sharing",
        action="store_true",
        default=False,
        help="Use weight sharing between token embeddings and output layer. Uses the Linear layer weight initialization.",
    )

    # validation generation
    parser.add_argument(
        "--val-prompt",
        type=str,
        default=VAL_PROMPTS[0],
        help="Prompt to use for validation generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate during text generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for text generation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k sampling for text generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p (nucleus) sampling for text generation",
    )

    return parser.parse_args()


def set_all_seeds(seed=42):
    logger.info(f"Setting all seeds to {seed}")
    # Python built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def load_tokenizer(tokenizer_pickle_path: str, special_tokens: list[str]):
    tokenizer = Tokenizer.from_pickle(tokenizer_pickle_path, special_tokens=special_tokens)
    logger.info(f"Tokenizer loaded from {tokenizer_pickle_path} with vocabulary size {len(tokenizer.vocab)}")
    return tokenizer


def load_model(args, vocab_size: int, device: str):
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        weight_sharing=args.weight_sharing,
        device=device,
    ).to(device)
    logger.info("TransformerLM specifications:")
    logger.info(f"  vocab_size: {vocab_size}")
    logger.info(f"  context length: {args.context_length}")
    logger.info(f"  num_layers: {args.num_layers}")
    logger.info(f"  num_heads: {args.num_heads}")
    logger.info(f"  d_model: {args.d_model}")
    logger.info(f"  d_ff: {args.d_ff}")
    logger.info(f"  rope_theta: {args.rope_theta}")

    logger.info("Calculating FLOPs...")
    with torch.no_grad():
        in_indices = torch.zeros(1, args.context_length, dtype=torch.int64)
        flops = FlopCountAnalysis(model, in_indices)
        logger.info(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # in billions

    logger.info(f"Number of parameters:\n{parameter_count_table(model)}")
    return model


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            logger.error("bf16 is not supported on this device. Please use fp16 or fp32.")
            sys.exit(1)
    else:
        precision = torch.float32

    logger.info(f"Using device: {device} and precision: {precision}")
    set_all_seeds(args.seed)
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

    # Initialize TensorBoard writer
    tb_logdir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_logdir)

    train_dataset = np.load(args.train_dataset, mmap_mode="r")
    val_dataset = np.load(args.val_dataset, mmap_mode="r")

    tokenizer = load_tokenizer(args.tokenized_dataset_pickle, args.special_tokens)
    vocab_size = len(tokenizer.vocab)
    model = load_model(args, vocab_size, device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        if not os.path.exists(args.resume_from_checkpoint):
            logger.error(f"Checkpoint path {args.resume_from_checkpoint} does not exist.")
            sys.exit(1)
        start_iter = load_checkpoint(args.resume_from_checkpoint, model, optimizer)
        logger.info(f"Resuming from iteration {start_iter}")
    else:
        start_iter = 0

    logger.info(f"Training for {args.max_train_iters} iterations")
    logger.info(f"Evaluation every {args.evaluation_iters} iterations")
    logger.info(f"Gradient clipping max norm: {args.gradient_max_norm}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Validation batch size: {args.val_batch_size}")

    train_dataloader = random_training_iterator(
        train_dataset, args.batch_size, args.context_length, device, args.max_train_iters
    )
    val_dataloader = SequentialValidationDataset(val_dataset, args.context_length, args.val_batch_size, device=device)

    for step, batch in (
        progress_bar := tqdm(
            enumerate(train_dataloader, start=start_iter),
            total=args.max_train_iters,
            desc="Training",
        )
    ):
        input_tokens, label_tokens = batch
        input_tokens = input_tokens.to(device)
        label_tokens = label_tokens.to(device)

        with autocast(device_type=device, dtype=precision):
            logits = model(input_tokens)  # (N,seq_len,vocab_size)
            loss = cross_entropy(logits, label_tokens)
        perplexity = torch.exp(loss)
        loss.backward()
        gradient_clipping(model.parameters(), args.gradient_max_norm)
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, args.lr_warmup_iters, args.max_train_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad()  # clear gradients before next loss.backward()
        logs = {
            "loss/train": loss.detach().item(),
            "perplexity/train": perplexity.detach().item(),
            "lr": lr,
        }
        progress_bar.set_postfix(**logs)
        writer.add_scalar("loss/train", loss.detach().item(), step)
        writer.add_scalar("perplexity/train", perplexity.detach().item(), step)
        writer.add_scalar("learning_rate", lr, step)

        if step > 0 and (step % args.checkpoint_iters == 0 or step == args.max_train_iters - 1):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            save_checkpoint(
                model,
                optimizer,
                step,
                checkpoint_path,
            )

        if step > 0 and (step % args.evaluation_iters == 0 or step == args.max_train_iters - 1):
            torch.cuda.empty_cache()
            with autocast(device_type=device, dtype=precision):
                val_metrics = run_validation(
                    model,
                    val_dataloader,
                    limit_val_iters=args.limit_val_iters,
                    global_step=step,
                    writer=writer,
                )
            val_print_str = (
                f"Validation metrics [Iteration {step}]: "
                f"loss = {val_metrics['loss/val']:.2f}, perplexity = {val_metrics['perplexity/val']:.2f}"
            )
            logger.info(val_print_str)
            run_generation(
                model,
                tokenizer,
                val_prompts=[args.val_prompt],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                device=device,
                global_step=step,
                writer=writer,
            )

    writer.close()
    return 0


def run_validation(
    model: TransformerLM,
    val_dataloader: SequentialValidationDataset,
    limit_val_iters: int = 0,
    global_step: int = 0,
    writer: SummaryWriter = None,
):
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for step, (input_tokens, label_tokens) in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader) if limit_val_iters == 0 else limit_val_iters,
            desc="Validation",
        ):
            if limit_val_iters > 0 and step >= limit_val_iters:
                break

            # AMP autocast
            logits = model(input_tokens)  # (N,seq_len,vocab_size)
            loss = cross_entropy(logits, label_tokens)
            total_loss += loss.detach().item()

    avg_loss = total_loss / (1 + step)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    val_metrics = {
        "loss/val": avg_loss,
        "perplexity/val": avg_perplexity,
    }
    for key, value in val_metrics.items():
        if writer is not None:
            writer.add_scalar(key, value, global_step)

    torch.cuda.empty_cache()
    model.train()
    return val_metrics


def run_generation(
    model: TransformerLM,
    tokenizer: Tokenizer,
    val_prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
    global_step: int = 0,
    writer: SummaryWriter = None,
):
    logger.info(f"Running generation with seed {seed}")
    for i, val_prompt in enumerate(val_prompts):
        generated_text = generateLLM(
            model,
            tokenizer,
            val_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            device=device,
        )
        print_text = f"PROMPT:\n{val_prompt}\nGENERATED:\n{val_prompt}{generated_text}"
        logger.info(f"\n{print_text}")
        writer.add_text(f"val_generations/{i}", print_text, global_step=global_step)


if __name__ == "__main__":
    args = get_args()
    sys.exit(train(args))
