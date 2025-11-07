from utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
import torch

# utils/eval_utils.py
def evaluate_model(model, eval_dataloader, device, global_rank):
    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses += loss.float()

    losses = losses / (step + 1)
    perplexity = torch.exp(losses) if not torch.isnan(losses) else float("inf")

    try:
        perplexity = get_all_reduce_mean(perplexity).item()
        loss = get_all_reduce_mean(losses).item()
    except:
        loss = float("inf")

    return loss, perplexity



        # try:
        #     perplexity = torch.exp(losses)
        # except OverflowError:
        #     perplexity = float("inf")

        # try:
        #     perplexity = get_all_reduce_mean(perplexity).item()
        # except Exception as e:
        #     print_rank_0(f"[WARNING] get_all_reduce_mean failed: {e}", args.global_rank)

        # try:
        #     loss = get_all_reduce_mean(losses).item()
        # except:
        #     loss = float("inf")

