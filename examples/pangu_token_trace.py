"""Print the generated PanGu token sequence after one offline vLLM request.

This is an offline trace: ``LLM.generate`` returns after decoding finishes, then
the script replays exactly the generated token IDs in their sampling order.
It deliberately does not change ModelRunner or the inference path.
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/w00939120/code/pangu/models",
        help="本地 PanGu 模型目录（包含 config.json、tokenizer.model 与 safetensors）。",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句话介绍昇腾 NPU。",
        help="要生成的输入文本。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="最多采样多少个输出 token。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # `use_fast=False` 对应 PanGu 仓库提供的 SentencePiece tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        tokenizer_mode="slow",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        disable_log_stats=True,
    )
    result = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )[0].outputs[0]

    generated_ids = list(result.token_ids)
    print("\n逐 token 回放（顺序与采样顺序一致）：")
    for step, token_id in enumerate(generated_ids, start=1):
        token_piece = tokenizer.convert_ids_to_tokens(token_id)
        text_so_far = tokenizer.decode(
            generated_ids[:step],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(
            f"第 {step:02d} 轮 | id={token_id:<6} "
            f"| 片段={token_piece!r} | 累计文本={text_so_far!r}")

    print(f"\n最终输出：{result.text!r}")


if __name__ == "__main__":
    main()
