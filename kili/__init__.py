"""kili: code for parakeet-tamil-kili (Tamil, English and code-switched ASR).

  kili.infer    NeMo inference, ONNX export, CPU runtime (no PyTorch), Home Assistant server
  kili.text     label normaliser, number words, convention folding, loanword engine
  kili.metrics  fold WER/CER, PER, EER
  kili.eval     short- and long-form evaluation
  kili.lm       KN 5-gram LM (Sangraha) and the gated beam decode
  kili.punct    Gemma 4 12B punctuation of training labels
  kili.synth    random-speaker synthetic voices (Maya2) and the label-free voice range
  kili.kernels  Blackwell training kernels (2.54x)
  kili.train    EMA, warmup-stable-decay LR, GPU augmentations
"""

__version__ = "0.1.0"
