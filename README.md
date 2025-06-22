# KokushiMD-10: Benchmark for Evaluating Large Language Models on Ten Japanese National Healthcare Licensing Examinations
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2506.11114)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/juniorliu95/KokushiMD-10)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/collect/KokushiMD-10)


## Usage
1. test your models with the prompts in `prompt.py`.
2. copy the output results to `results/` with exactly the same format.
3. run `calculate_scores.py`, you will get all the scoring results in `scoring`.

## Structure
```
KokushiMD_eval/
├── calculate_scores.py          # Main scoring script for evaluating LLM results
├── utils.py                     # Utility functions and constants
├── exams/                       # Examination data directory
│   └── JA/                      # Japanese examination data
├── results/                     # LLM evaluation results
│   └── [company]/               # Results organized by company
│       └── [model]/             # Results organized by model
│           └── [input_type]/    # Results by input type (text/multimodal)
│               └── [exam_type]/ # Results by input type (text/multimodal)
└── scoring/                     # Scoring results directory containing eval results
    └── [company]/               # Scores organized by company
        └── [model]/             # Scores organized by model
            └── [input_type]/    # Scores by input type
                └── [exam_type]/ # Scores by healthcare profession
                    └── [score_files]

```


## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{liu2025kokushimd10,
  title={KokushiMD-10: Benchmark for Evaluating Large Language Models on Ten Japanese National Healthcare Licensing Examinations},
  author={Liu, Junyu and Yan, Kaiqi and Wang, Tianyang and Niu, Qian and Nagai-Tanima, Momoko and Aoyama, Tomoki},
  journal={arXiv preprint arXiv:2506.11114},
  year={2025}
}
```

## License

This dataset is released under the MIT License. See [LICENSE](LICENSE) for details.

## Data Source

The dataset is constructed from official Japanese national healthcare licensing examinations published by the Ministry of Health, Labour and Welfare of Japan between 2020-2024.

## Ethical Considerations

- All questions are from publicly available official examinations
- No patient privacy concerns as images are educational/examination materials
- Dataset intended for research and educational purposes
- Should not be used as a substitute for professional medical advice

<!-- ## Contact

For questions about the dataset or research collaboration:
- **Primary Contact**: Tomoki Aoyama (aoyama.tomoki.4e@kyoto-u.ac.jp)
- **Institution**: Kyoto University -->

## Acknowledgments

We thank the Ministry of Health, Labour and Welfare of Japan for making the examination materials publicly available, and all healthcare professionals who contributed to the creation of these rigorous assessments.

---

**Disclaimer**: This dataset is for research and educational purposes only. It should not be used for clinical decision-making or as a substitute for professional medical judgment.