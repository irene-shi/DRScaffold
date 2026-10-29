# DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language Models

[Xinrui Shi](https://irene-shi.github.io)\*, [Kai Liu](https://kai-liu.cn/)\*, Ziqing Zhang, Jianze Li, Anqi Li, and [Yulun Zhang](http://yulunzhang.com/), "DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language Models", NeurIPS, 2026

<div>
<a href="https://arxiv.org/abs/2605.26038" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv Paper"></a>
<a href="https://github.com/irene-shi/DRScaffold/raw/refs/heads/main/docs/DRScaffold-Supplementary.pdf" style="text-decoration: none;"><img src="https://img.shields.io/badge/GitHub-Supplementary-ffd21e?logo=github&amp;logoColor=white" alt="GitHub Supplementary"></a>
<a href="https://github.com/irene-shi/DRScaffold" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=irene-shi/DRScaffold"></a>
<a href="https://github.com/irene-shi/DRScaffold/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/irene-shi/DRScaffold?style=social"></a>
</div>

#### 🔥🔥🔥 News

- **2026-10-08:** Supplementary material, data construction code and training code are released. 
- **2026-09-25:** DRScaffold is accepted by NeurIPS 26! 🎉🎉🎉
- **2026-05-06:** This repo is released. 

---
![](figs/teaser.png)

> **Abstract:** Lightweight vision-language models perform competitively on standard benchmarks yet fail systematically in dense-scene reasoning, where multiple objects, attributes, and relations must be jointly grounded and resolved through multi-step inference. Such capability is critical for real-world applications where models must reliably interpret cluttered environments. Yet existing training signals provide no explicit grounding between reasoning steps and the underlying visual entities and relations, leaving lightweight models free to generate fluent but visually unanchored reasoning chains. To address this gap, we first introduce **DRBench**, a benchmark of 14,573 questions across 2,943 images, organized into five task categories spanning three progressive reasoning layers. Building on DRBench, we propose **DRScaffold**, a supervised fine-tuning framework that decomposes the supervision target into four causally ordered stages, promoting grounded reasoning without architectural modification. Experiments on three lightweight VLMs demonstrate substantial gains on DRBench while preserving or improving performance on general-purpose benchmarks. Notably, Qwen2.5-VL-3B trained with DRScaffold surpasses the frozen Qwen2.5-VL-32B under zero-shot direct-answer prompting on DRBench, demonstrating that structured supervision can substitute for a significant portion of model scale in dense-scene reasoning. 


## ⚒️ TODO
- [x] Release paper.
- [x] Release supplementary material.
- [x] Release data construction code.
- [x] Release training code.

## <a name="contents"></a> 🔗 Contents
- [DRBench](#drbench)
- [DRScaffold](#drscaffold)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## <a name="drbench"></a> 📊 DRBench

The [data construction code](data/README.md) implements question generation,
text-blind filtering, GT-supported scene review, human-exemplar annotation,
consistency revision, and training export. Configure your own models and API
credentials and supply your own human examples.

- data construction pipeline
  <p align="center">
    <img width="950" src="figs/drbench-pipeline.png">
  </p>

- data annotation sample
  <p align="center">
    <img width="950" src="figs/drbench-sample.png">
  </p>

- data distribution
  <p align="center">
    <img width="950" src="figs/drbench-pie.png">
  </p>

## <a name="drscaffold"></a> 💡 DRScaffold

DRScaffold decomposes supervision into four causally ordered stages: **Entity Grounding**, **Relation Modeling**, **Stepwise Reasoning**, and **Answer Generation**. The framework uses staged gradient masking to enforce grounded reasoning without modifying model architecture.

The [training code](train/README.md) supports Qwen2.5-VL-3B, InternVL2.5-2B,
and Phi-4-multimodal, with four-stage LoRA fine-tuning, progressive loss masking,
LLaVA replay mixing, and the paper’s stage-removal and ordering ablations.

<p align="center">
  <img width="950" src="figs/drscaffold.png">
</p>

## <a name="results"></a> 🔎 Results

DRScaffold consistently improves dense-scene reasoning quality over base lightweight VLMs while preserving performance on general-purpose benchmarks.

- quantitative comparisons
  <p align="center">
  <img width="950" src="figs/results-main.png">
  </p>

  <p align="center">
  <img width="950" src="figs/results-method.png">
  </p>

- qualitative comparisons

  <p align="center">
  <img width="950" src="figs/results-qualitative.png">
  </p>

  <p align="center">
  <img width="950" src="figs/results-stage.png">
  </p>

## <a name="citation"></a> 📎 Citation
If you find the code helpful in your research or work, please cite the following papers.

```
@article{shi2026drscaffold,
  title={DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language Models},
  author={Shi, Xinrui and Liu, Kai and Zhang, Ziqing and Li, Jianze and Li, Anqi and Zhang, Yulun},
  journal={arXiv preprint arXiv:2605.26038},
  year={2026}
}
```

## <a name="acknowledgements"></a> 💖 Acknowledgements

This project is based on [ms-swift](https://github.com/modelscope/ms-swift). Thanks for their great work.

## <a name="license"></a> 📄 License

The code in this repository is released under the [MIT License](LICENSE).
Third-party datasets, images, model weights, and dependencies remain subject to
their respective licenses.
