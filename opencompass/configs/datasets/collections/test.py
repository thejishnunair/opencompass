import os

from mmengine import read_base

with read_base():
  # from ..gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
  # from ..mmlu.mmlu_gen_4d595a import mmlu_datasets
  # from ..mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets
  # from ..gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
  # from ..hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
  # from ..drop.drop_openai_simple_evals_gen_3857b0 import drop_datasets
  # from ..aime2024.aime2024_gen_6e39a4 import aime2024_datasets
  # from ..math.math_prm800k_500_0shot_cot_gen import (
  #     math_datasets,
  # )
  # from ..IFEval.IFEval_gen_3321a3 import ifeval_datasets
  # from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import (
  #   LCBCodeGeneration_dataset,
  # )
  from ..humaneval.humaneval_openai_sample_evals_gen_dcae0e import humaneval_datasets
  # from opencompass.configs.datasets.subjective.multiround.mtbench101_judge_new import \
  #     mtbench101_datasets  # noqa: F401, E501
  # from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_new import \
  #     wildbench_datasets


datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')), []
) # + [LCBCodeGeneration_dataset]

# datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')
#                 and 'mtbench101' not in k and 'wildbench' not in k), [])
# datasets += mtbench101_datasets  # noqa: F401, E501
# datasets += wildbench_datasets

core_summary_groups = [
    {
        'name': 'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['bbh', 'naive_average'],
            ['math_prm800k_500', 'accuracy'],
            ['aime2024', 'accuracy'],
            ['GPQA_diamond', 'accuracy'],
            ['mmlu_pro', 'accuracy'],
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