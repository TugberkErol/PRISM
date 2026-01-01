PRISM: Past-Regularized Iterative Self-Distillation With Momentum for Polyp Segmentation

Abstract
Polyps are abnormal tissue growths in the colon that may develop into colorectal cancer if left undetected. Accurate segmentation in medical imaging is essential for early diagnosis and treatment. Although deep learning has greatly improved polyp segmentation, its dependence on large annotated datasets and substantial computational resources hampers generalization across diverse clinical settings. To overcome these challenges, we propose PRISM, a momentum-based self-distillation method that improves segmentation performance without introducing additional inference cost. Instead of storing or reusing past predictions, PRISM constructs a temporally smoothed teacher model by applying an exponential moving average (EMA) to the model's weights throughout training. This momentum-based teacher provides stable and adaptive supervision signals that co-evolve with the student model. PRISM achieves a Dice score of 0.81 and an IoU of 0.75, outperforming baseline and conventional self-distillation methods.

Paper Link
The full article is published in Healthcare Technology Letters (2025): https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/htl2.70050

Citation
If you find this work useful in your research, please cite our paper:


@article{erol2025prism,
  title   = {PRISM: Past-Regularized Iterative Self-Distillation With Momentum for Polyp Segmentation},
  author  = {Erol, Tugberk and Caglikantar, Tuba and Sarikaya, Duygu},
  journal = {Healthcare Technology Letters},
  year    = {2025},
  volume  = {12},
  pages   = {070050},
  doi     = {10.1049/htl2.70050}
}
