"""PowerShell: python -m mini_engine --model D:\\models\\Qwen3-0.6B --prompt hello"""

import argparse
import json
from dataclasses import asdict

from .cli import add_runner_arguments, runner_config_from_args
from .model import NativeQwen3Runner


def main():
    parser = argparse.ArgumentParser(description="Single-request native Qwen3 inference")
    add_runner_arguments(parser)
    parser.add_argument("--prompt")
    parser.add_argument("--chat", action="store_true", help="start an interactive multi-turn chat")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--speculative", choices=["none", "prompt_lookup"], default="none")
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--adaptive-draft", action="store_true")
    args = parser.parse_args()
    if args.cuda_graph and args.speculative != "none":
        parser.error("--cuda-graph and --speculative cannot be combined yet")
    if not args.chat and args.prompt is None:
        parser.error("--prompt is required unless --chat is used")
    if args.max_new_tokens <= 0 or args.max_draft_tokens <= 0:
        parser.error("token limits must be positive")
    try:
        config = runner_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    runner = NativeQwen3Runner.from_pretrained(config.model_name_or_path, **config.runner_kwargs())
    try:
        if args.chat:
            messages = []
            print("MiniQwenEngine chat; type /exit to quit, /clear to reset.")
            while True:
                try:
                    user_text = input("you> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if user_text.strip() == "/exit":
                    break
                if user_text.strip() == "/clear":
                    messages = []
                    print("conversation cleared")
                    continue
                if not user_text.strip():
                    continue
                messages.append({"role": "user", "content": user_text})
                if args.speculative == "prompt_lookup":
                    result = runner.generate_speculative(messages=messages, max_new_tokens=args.max_new_tokens,
                        max_draft_tokens=args.max_draft_tokens, adaptive_draft=args.adaptive_draft).text
                else:
                    result = runner.chat(messages, args.max_new_tokens, use_cuda_graph=args.cuda_graph)
                messages.append({"role": "assistant", "content": result})
                print(f"assistant> {result}")
        else:
            if args.prompt is None:
                parser.error("--prompt is required unless --chat is used")
            if args.speculative == "prompt_lookup":
                result = runner.generate_speculative(args.prompt, args.max_new_tokens,
                    max_draft_tokens=args.max_draft_tokens, adaptive_draft=args.adaptive_draft)
            else:
                result = runner.generate_incremental(args.prompt, args.max_new_tokens, use_cuda_graph=args.cuda_graph)
            payload = asdict(result)
            payload["quantization"] = runner.model.quantization_report
            payload["backends"] = runner.model.backend_report()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
