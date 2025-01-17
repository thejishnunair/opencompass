import os

from mmengine import read_base

# enable_gsm8k = os.getenv("ENABLE_GSM8K", "false").lower() == "true"
# enable_mmlu = os.getenv("ENABLE_MMLU", "true").lower() == "true"
# enable_drop = os.getenv("ENABLE_DROP", "false").lower() == "true"
# enable_cmmlu = os.getenv("ENABLE_CMMLU", "false").lower() == "true"
# enable_mmlu_pro = os.getenv("ENABLE_MMLU_PRO", "false").lower() == "true"
# enable_gpqa = os.getenv("ENABLE_GPQA", "false").lower() == "true"
# enable_hellaswag = os.getenv("ENABLE_HELLASWAG", "false").lower() == "true"
# enable_aime = os.getenv("ENABLE_AIME", "false").lower() == "true"
# enable_math = os.getenv("ENABLE_MATH", "false").lower() == "true"
# enable_humaneval = os.getenv("ENABLE_HUMANEVAL", "false").lower() == "true"
# enable_livecodebench = os.getenv("ENABLE_LIVECODEBENCH", "false").lower() == "true"
# enable_ifeval = os.getenv("ENABLE_IFEVAL", "false").lower() == "true"
# enable_alpacaeval = os.getenv("ENABLE_ALPACAEVAL", "false").lower() == "true"
# enable_mtbench101 = os.getenv("ENABLE_MTBENCH101", "false").lower() == "true"
# enable_wildbench = os.getenv("ENABLE_WILDBENCH", "false").lower() == "true"


with read_base():
  from ..gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
  from ..mmlu.mmlu_gen_4d595a import mmlu_datasets
  from ..mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
  from ..gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
  from ..hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
  from ..drop.drop_openai_simple_evals_gen_3857b0 import drop_datasets
  from ..aime2024.aime2024_gen_6e39a4 import aime2024_datasets
  from ..math.math_prm800k_500_0shot_cot_gen import (
      math_datasets,
  )
  from ..IFEval.IFEval_gen_3321a3 import ifeval_datasets
  from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import (
    LCBCodeGeneration_dataset,
  )
  #from ..humaneval.humaneval_openai_sample_evals_gen_dcae0e import humaneval_datasets

#     pass
    # if enable_gsm8k:
    #     from ..gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # if enable_cmmlu:
    #     from ..cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    # if enable_mmlu_pro:
    #     from ..mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
    # if enable_gpqa:
    #     from ..gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
    # if enable_hellaswag:
    #     from ..hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
    # if enable_drop:
    #     from ..drop.drop_openai_simple_evals_gen_3857b0 import drop_datasets
    # if enable_aime:
    #  from ..aime2024.aime2024_gen_6e39a4 import aime2024_datasets
    # if enable_math:
    #     from ..math.math_gen_265cce import math_datasets
    # if enable_humaneval:
    #     from ..humaneval.humaneval_gen_8e312c import humaneval_datasets
    # if enable_livecodebench:
    #     from ..livecodebench.livecodebench_gen_6966bc import LCB_datasets
    # if enable_ifeval:
    #     from ..IFEval.IFEval_gen_3321a3 import ifeval_datasets
    # if enable_alpacaeval:
    #     from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4_bradleyterry import (
    #         alpacav2_datasets,
    #     )
    # if enable_mtbench101:
    #     from opencompass.configs.datasets.subjective.multiround.mtbench101_judge_new import \
    #         mtbench101_datasets
    # if enable_wildbench:
    #     from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_new import \
    #         wildbench_datasets




datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')), []
) + [LCBCodeGeneration_dataset]

core_summary_groups = [
    {
        'name': 'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['bbh', 'naive_average'],
            ['math_prm800k_500', 'accuracy'],
            ['aime2024', 'accuracy'],
            ['GPQA_diamond', 'accuracy'],
            ['mmlu_pro', 'naive_average'],
            ['openai_humaneval', 'humaneval_pass@1'],
            ['lcb_code_generation', 'pass@1'],
        ],
    },
]


summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'General Reasoning',
        ['GPQA_diamond', 'accuracy'],
        '',
        'Math Calculation',
        ['math_prm800k_500', 'accuracy'],
        ['aime2024', 'accuracy'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['lcb_code_generation', 'pass@1'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)