# parakeet-tamil-kili · தமிழ் கிளி

Speech recognition for Tamil, English, and the two mixed together, small enough to run on a CPU.
A 0.6B FastConformer-TDT fine-tuned from [NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

- **Write-up** (interactive, English and Tamil): https://shrirajh.github.io/parakeet-tamil-kili/
- **Try it in your browser** (CPU, nothing leaves your device): https://shrirajh.github.io/parakeet-tamil-kili/try/
- **Model**: https://huggingface.co/shrirajh/parakeet-tamil-kili
- **Code**: https://github.com/shrirajh/parakeet-tamil-kili

## Results

Fold WER, %; lower is better. 5 epochs of run 10, greedy decoding; "+ LM" is the gated beam-4 decode with the
Sangraha 5-gram. Up to 2,000 clips per test set (all of FLEURS: 580, MILE: 1,333). PER and EER for every set are in
the write-up.

| test set | kind | **kili** | kili + LM | indic-transcribe-core | IndicWhisper | parakeet v3 |
|---|---|---|---|---|---|---|
| IndicVoices | conversational | **34.05** | 33.53 | 49.28 | 68.84 | 103.92 |
| Kathbath | read | 29.15 | 25.38 | 20.80 | 21.77 | 117.16 |
| MILE | read | 27.89 | 26.31 | 23.37 | 19.61 | 114.87 |
| FLEURS ta | read | 30.46 | 27.36 | 21.47 | 23.25 | 114.78 |
| Common Voice ta | read | 35.22 | 32.60 | 26.70 | 29.73 | 111.26 |
| OpenSLR-65 | read | 24.84 | 20.73 | 15.19 | 24.43 | 114.76 |
| Common Voice en | English | 10.19 | 10.19 | 11.94 | – | 9.25 |
| LibriSpeech clean / other | English | 2.30 / 4.07 | same | 2.15 / 3.88 | – | 2.17 / 3.51 |
| Svarah | Indian-accented English | 10.67 | 10.65 | 9.74 | – | 9.62 |

What this says:

- **Conversational Tamil** is where kili leads, by a wide margin (15 WER over the next best open model).
- **Read Tamil** (news, prompts) is behind indic-transcribe-core and IndicWhisper. The gap is mostly acoustic: PER
  (sounds) differs far less than WER (spellings).
- **English** stays close to the parakeet v3 it started from. It does not forget English.
- **Code-switching** is the default. English words come out in Latin script with the Tamil suffix attached
  (`meeting-க்கு`, `office-ல`): 89% of the English words in conversational Tamil (IndicVoices), 6% in Tamil script.
  English speech stays English: Tamil script appears in 0.1–0.2% of LibriSpeech and Common Voice outputs. On
  Indian-accented English (Svarah) it is 9.5% of clips but 1.5% of words, mostly places and names that have Tamil
  spellings (டெல்லி, லண்டன், ஜப்பான்), Indian-language words and one-word clips; the meaning is kept. (The loanword
  converter's recall of 0.64 is measured on the training labels, not on the model's output: converting the common
  loanwords consistently taught the model the rule, and it applies it to most of the loanwords the converter left alone.)

**CPU speed**: the int8 ONNX encoder runs about 28× faster than real time on 8 threads, with a peak of about 2.8 GB
of memory. No PyTorch is needed.

### Metrics

- **fold WER / CER**: WER and CER after folding writing conventions on both sides. Latin vs Tamil-script loanwords,
  number words, and punctuation and case all fold. Plain WER counts `மீட்டிங்` vs `meeting` as an error;
  fold WER does not. (`kili.metrics.wer`)
- **PER**: a feature-weighted Tamil phoneme edit distance. Sounds that speakers merge (ல/ள/ழ, ண/ந/ன, ர/ற, short
  and long vowels) cost 0.25, close neighbours 0.5, more distant relatives 0.75, anything else 1. PER separates
  "misheard" from "misspelled". (`kili.metrics.per`)
- **EER**: embedding error rate, 1 − cosine between EmbeddingGemma 2 sentence embeddings of reference and output.
  Against blind meaning judgements it separates same-meaning from wrong-meaning outputs at AUC 0.88.
  (`kili.metrics.eer`)

## Use

The code is one package, `kili`. Install what you need with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/shrirajh/parakeet-tamil-kili && cd parakeet-tamil-kili
uv sync --extra nemo      # NeMo inference / evaluation (GPU or CPU)
uv sync --extra onnx      # CPU runtime only, no PyTorch
uv sync --extra wyoming   # Home Assistant server
```

**NeMo** (GPU or CPU):

```bash
uv run python -m kili.infer.transcribe --model shrirajh/parakeet-tamil-kili clip.wav
```

```python
from kili.infer.transcribe import load_model
model = load_model("shrirajh/parakeet-tamil-kili")   # sets local attention [256, 256], as trained
print(model.transcribe(["clip.wav"])[0].text)
```

The model was trained with local attention and a ±256-frame window (±20.48 s). Load it with `load_model`, or call
`model.change_attention_model("rel_pos_local_attn", [256, 256])` yourself. NeMo's long-audio default of
[128, 128] costs about 3.6 WER. With local attention, memory grows linearly with length, so long recordings are decoded whole.

**CPU, ONNX** (no PyTorch):

```bash
uvx hf download shrirajh/parakeet-tamil-kili --include "onnx/*" --local-dir kili
uv run python -m kili.infer.onnx_run --dir kili/onnx clip.wav
```

`onnx/` holds the int8 encoder (0.9 GB, the default) and the fp32 encoder (2.4 GB, `--precision fp32`), with the
prediction network and joint as numpy weights (`dec.npz`). The fp32 export transcribes identically to NeMo; int8
costs at most about 0.3 WER.

**Home Assistant** (Wyoming speech-to-text, CPU):

```bash
uv run python -m kili.infer.wyoming_server --dir kili/onnx --uri tcp://0.0.0.0:10300
```

**Language model** (optional, `lm/` in the model repo): this is a token-level KN 5-gram over 8M Sangraha sentences,
with every test-set sentence removed. A gate keeps the plain greedy output for clips that are more than half Latin
letters, so English is untouched. On read Tamil it fixes spellings; it cannot fix what the acoustic model misheard.
It also drops commas.

```bash
uvx hf download shrirajh/parakeet-tamil-kili --include "lm/*" --local-dir kili
uv run python -m kili.infer.transcribe --model shrirajh/parakeet-tamil-kili --lm kili/lm/sangraha_o5.nemo clip.wav
```

## Training

- **Start**: parakeet-tdt-0.6b-v3, with 2,044 Tamil-script pieces appended to its 8,192 (`kili.tokenizer`). English
  tokenisation and weights are unchanged, and a Tamil sentence takes 27 pieces instead of 57. Each new piece is
  warm-started from the mean of v3's rows for its romanised spelling (க்கு starts as "kku"), so after the surgery
  the model scores exactly like stock v3.
- **Data**: 6,534 hours, of which:
  - Tamil: IndicVoices, Shrutilipi, Kathbath, SPRING-INX, YODAS v3 Tamil (relabelled), and 525 h of synthetic
    Tamil-English;
  - English: a slice of Granary as a replay set.
  
  The data table with hours, licences and how often each source was seen is in the write-up and
  [`configs/train_r10.yaml`](https://github.com/shrirajh/parakeet-tamil-kili/blob/main/configs/train_r10.yaml).
- **Labels**:
  - English loanwords are rewritten into Latin script by a loanword engine (`kili.text.loanwords`, 23,804 Tamil-script spellings of English words).
  - Numbers are written as Tamil words.
  - Punctuation was added with Gemma 4 12B, which may only add marks, never change letters (`kili.punct`).
  - About 31k recordings with broken or duplicate labels were removed.
- **Synthetic speech**:
  - Maya2 TTS with random speaker vectors, so nobody's voice is cloned.
  - The voices are spread along a pitch direction learned from synthetic audio alone, with no gender labels
    (`kili.synth`).
- **Recipe**:
  - Warmup-stable-decay learning rate with decay forks, plus EMA.
  - GPU augmentation: noise, reverb, phone line, codecs, babble, overlapping talkers, speed ±10%.
  - 5 epochs on 8× RTX PRO 6000 Blackwell. A 6th epoch made read Tamil worse, so the release is the 5-epoch fork.
- **Kernels**: custom Triton kernels took training from 358 to 911 audio-hours per GPU-hour (2.54×):
  - banded rel-pos attention;
  - depthwise conv;
  - fused residual + dropout + LayerNorm;
  - TDT loss row kernels.
  
  Each is exactness-tested against NeMo (`kili.kernels`). The whole project cost about USD 400 of rented GPUs.

## Files on Hugging Face

```
parakeet-tamil-kili.nemo   NeMo model (weights, tokenizer, config)
model.safetensors          the same weights, config.yaml and tokenizer/ beside it
onnx/                      CPU export: int8 and fp32 encoders, numpy decoder weights, front end, meta.json
lm/sangraha_o5.nemo        optional 5-gram LM (NeMo NGPU-LM format)
SHA256SUMS
```

## Repository

```
kili/infer     NeMo inference, ONNX export, CPU runtime (no PyTorch), Wyoming server
kili/tokenizer Tamil tokens for parakeet v3: tokenizer extension, model surgery, sound-based warm start
kili/text      label normaliser, Tamil and English number words, convention folding, loanword engine
kili/metrics   fold WER/CER, PER, EER
kili/eval      short- and long-form evaluation
kili/lm        Sangraha corpus with test-set exclusion, KN 5-gram (ARPA + NGPU-LM), gated beam decode
kili/punct     Gemma 4 12B punctuation prompts and runner (vLLM, OpenAI-compatible)
kili/synth     random-speaker synthetic voices and the voice-range direction
kili/kernels   Blackwell training kernels for NeMo FastConformer-TDT
kili/train     EMA, warmup-stable-decay schedule, GPU augmentations
configs/       run 10 training recipe
docs/          the write-up (GitHub Pages)
```

Tests: `uv run --group dev pytest`. The GPU, NeMo and ONNX tests skip unless those are installed. Point them at a
model with `KILI_TEST_MODEL`, `KILI_ONNX_DIR`, `KILI_TEST_LM` and `KILI_TEST_AUDIO`.

## Licences

- **Model weights**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Built on NVIDIA
  parakeet-tdt-0.6b-v3 (CC BY 4.0).
- **Code**: [Apache-2.0](https://github.com/shrirajh/parakeet-tamil-kili/blob/main/LICENSE).
- **Language model**: CC BY 4.0. It is derived from Sangraha (AI4Bharat, CC BY 4.0).
- **Transcripts** you make with the model are not covered by either licence.

**One request** (not a licence condition). Use it for anything. In return, all I ask is that you give something
back: share what you learn, keep knowledge open, teach someone, and don't use it to harm people.

Training data: IndicVoices, Shrutilipi, SPRING-INX (CC BY 4.0); Kathbath; YODAS and Granary (CC BY 3.0); synthetic
speech from Maya2 (Apache-2.0) with Kyutai Mimi (CC BY 4.0) on text from Tanglish-Corpus-185k (CC BY 4.0). The
punctuation few-shot examples in `kili/punct/prompts.py` are IndicVoices transcripts (CC BY 4.0). Full credits are
in the write-up.

Built by shrirajh with [Claude](https://claude.ai) as a coding and research assistant.
