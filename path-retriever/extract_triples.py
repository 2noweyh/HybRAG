# extract_triples.py
import argparse, torch
from trainer_kbqa import Trainer_KBQA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["webqsp", "cwq"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    trainer = Trainer_KBQA.load_from_checkpoint(args.checkpoint)  # or 직접 init
    if args.split == "test":
        data = trainer.test_data
    else:
        data = trainer.valid_data

    trainer.extract_topk_triples(data, k=args.topk, split=args.split, save_path=args.save_path)

if __name__ == "__main__":
    main()
