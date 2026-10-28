"""Windows-native chat server entry point and compatible public imports.

Protocol validation, the inference worker, and HTTP transport live in
``mini_engine.serving``. ``python -m mini_engine.server`` remains supported.
"""

import argparse
import math

from .cli import add_runner_arguments, runner_config_from_args

from .serving import APIError, ChatHTTPServer, InferenceService

__all__ = ["APIError", "ChatHTTPServer", "InferenceService", "main"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runner_arguments(parser, serving=True)
    parser.add_argument("--served-model-name", default="qwen3-0.6b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    if (not args.host or not 0 <= args.port <= 65535 or args.max_pending <= 0
            or args.max_new_tokens <= 0 or args.prefill_chunk_size <= 0
            or args.max_num_batched_tokens <= 0
            or (args.max_context is not None and args.max_context <= 0)
            or not math.isfinite(args.request_timeout) or args.request_timeout <= 0):
        parser.error("service limits must be positive and port must be in [0, 65535]")
    try:
        config = runner_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    from .model import NativeQwen3Runner
    runner = NativeQwen3Runner.from_pretrained(config.model_name_or_path, **config.runner_kwargs())
    service = server = None
    try:
        service = InferenceService(runner, model_name=args.served_model_name, max_pending=args.max_pending,
            max_context=args.max_context, max_new_tokens=args.max_new_tokens,
            request_timeout=args.request_timeout, use_cuda_graph=args.cuda_graph,
            enable_prefix_cache=not args.disable_prefix_cache, prefill_chunk_size=args.prefill_chunk_size,
            max_num_batched_tokens=args.max_num_batched_tokens)
        service.start()
        server = ChatHTTPServer((args.host, args.port), service)
        print(f"mini-engine listening on http://{args.host}:{server.server_port}; greedy decoding")
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            if service is not None:
                service.close()
            # A failed join must not release buffers still used by the worker.
            runner.close()


if __name__ == "__main__":
    main()
