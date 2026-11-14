# parakeet-tamil-kili · தமிழ் கிளி

The leading open speech recognition model for conversational Tamil, including Tamil mixed with English. It is
fast (about 28× real time on 8 CPU threads) and competitive on read Tamil.
A 0.6B FastConformer-TDT fine-tuned from [NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
We chose a transducer because it writes as it listens, frame by frame, which pushes it to write Tamil as it is
spoken (pronunciation, fillers, repetitions, backchannels) rather than as it would be written; it also gives
word timestamps.

- **Write-up** (interactive, English and Tamil): https://shrirajh.github.io/parakeet-tamil-kili/
- **Try it in your browser** (CPU, nothing leaves your device): https://shrirajh.github.io/parakeet-tamil-kili/try/
- **Model**: https://huggingface.co/shrirajh/parakeet-tamil-kili
- **Code**: https://github.com/shrirajh/parakeet-tamil-kili

## Results

Tamil: PER, sounds misheard (%). English: WER (%). Lower is better; best per row in bold. 5 epochs of run 10, greedy
decoding; "+ LM" is the gated beam-4 decode with the Sangraha 5-gram. Up to 2,000 clips per test set (all of FLEURS:
580, MILE: 1,333), the same clips and the same scorer for every model.

| test set | kind | **[kili](https://huggingface.co/shrirajh/parakeet-tamil-kili)** | [kili + LM](https://huggingface.co/shrirajh/parakeet-tamil-kili/tree/main/lm) | [indic-transcribe-core](https://huggingface.co/bodhan-ai/indic-transcribe-core) | [Hima-ASR](https://huggingface.co/Prahaladha/hima-asr) | [Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) | [IndicWhisper](https://github.com/AI4Bharat/vistaar) | [original parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
|---|---|---|---|---|---|---|---|---|
| *parameters* | | *0.63B* | *0.63B + LM* | *1.22B* | *0.46B* | *33.7B (4.3B active)* | *0.77B* | *0.63B* |
| *word timestamps* | | *yes, natively* | *no* | *no* | *no* | *yes, natively* | *approximately* | *yes, natively* |
| IndicVoices | conversational | **9.44** | 9.89 | 11.52 | 12.39 | 14.08 | 23.83 | 65.59 |
| Long recordings (2–5 min) | long-form | **7.39** | – | 7.67 | 7.55 | 15.43 | – | – |
| Kathbath | read | 2.41 | 2.17 | **1.33** | 1.46 | 1.62 | 1.57 | 33.50 |
| MILE | read | 2.08 | 2.12 | 1.54 | **1.41** | 1.43 | **1.41** | 32.21 |
| FLEURS ta | read | 6.66 | 6.40 | 5.27 | **4.87** | 5.72* | 5.58 | 37.00 |
| Common Voice ta | read | 2.86 | 2.79 | **1.62** | 1.69 | 1.58* | 2.33 | 27.91 |
| OpenSLR-65 | read | 1.58 | 1.36 | **0.69** | 0.96 | 0.73 | 13.24 | 29.00 |
| Common Voice en | English | 10.19 | 10.19 | 11.94 | 20.08 | 8.40* | – | **9.25** |
| LibriSpeech clean / other | English | 2.30 / 4.07 | same | 2.15 / 3.88 | 9.61 / 12.26 | **1.68 / 2.92** | – | 2.17 / 3.51 |
| Svarah | Indian-accented English | 10.67 | 10.65 | 9.74 | 11.98 | **8.87** | – | 9.62 |

Meaning (Tamil EER, %): IndicVoices kili **2.66**, indic-transcribe-core 3.44, Hima-ASR 3.69, Nemotron 4.44. On read
Tamil, kili is at 1.4–2.5 and the best model on each set at 0.5–1.6.

<details>
<summary>Tamil fold WER / CER (%)</summary>

| test set | kili | kili + LM | indic-transcribe-core | Hima-ASR | Nemotron 3.5 | IndicWhisper | original parakeet v3 |
|---|---|---|---|---|---|---|---|
| IndicVoices | **34.05** / **11.62** | 33.53 / 11.93 | 49.28 / 14.27 | 50.88 / 15.56 | 54.07 / 17.52 | 68.84 / 31.50 | 103.92 / 80.43 |
| Long recordings | **34.37** / 10.44 | – | 38.71 / **10.17** | 39.18 / 10.40 | 50.34 / 21.95 | – | – |
| Kathbath | 29.15 / 5.56 | 25.38 / 4.85 | **20.80** / **3.56** | 21.21 / 3.65 | 21.06 / 3.83 | 21.77 / 3.82 | 117.16 / 60.46 |
| MILE | 27.89 / 5.02 | 26.31 / 4.91 | 23.37 / 4.00 | 19.04 / 3.20 | **16.28** / **2.89** | 19.61 / 3.18 | 114.87 / 59.76 |
| FLEURS ta | 30.46 / 9.71 | 27.36 / 9.09 | **21.47** / 7.43 | 22.53 / **7.34** | 23.05* / 8.10* | 23.25 / 7.85 | 114.78 / 61.83 |
| Common Voice ta | 35.22 / 6.37 | 32.60 / 5.93 | **26.70** / 4.21 | **26.70** / **4.13** | 13.30* / 2.70* | 29.73 / 5.20 | 111.26 / 56.23 |
| OpenSLR-65 | 24.84 / 4.62 | 20.73 / 3.83 | 15.19 / 2.50 | 17.77 / 3.02 | **6.97** / **1.47** | 24.43 / 14.78 | 114.76 / 58.36 |

</details>

\* Nemotron 3.5 Transcribe's training data includes Common Voice and FLEURS, so these scores may include recordings
it trained on; they are not counted as best. Its OpenSLR-65 score is also unusually low. Hima-ASR's model card does
not say what it was trained on. Both were run with their own code at fixed versions (Hima-ASR told the language,
Nemotron with the prompts from its model card). Word timestamps: indic-transcribe-core's vocabulary has timestamp tokens, but its released code turns them
off; IndicWhisper's come from Whisper's attention alignment; kili + LM's beam decode does not return them. Long recordings are cut into pieces of up to 30 s for every model;
given whole recordings, Nemotron wrote 8 of the 20 in romanised Tamil.

What this says:

- **Conversational Tamil** is where kili leads every open model we tested, including one over 50× its size:
  18% fewer sound errors than the next best on IndicVoices (PER 9.4 vs 11.5) and the best PER on long recordings.
- **Read Tamil** (news, prompts) is competitive: 0.7 to 1.8 PER points behind the best. The gap is mostly spelling, not
  hearing: PER (sounds) differs far less than WER (spellings).
- **It writes what was said, and does not editorialise.** In 2,000 everyday recordings the three strongest other
  models rewrote spoken Tamil as written Tamil about half the time (நாங்க → நாங்கள், இல்ல → இல்லை: 53–54% of
  976 cases); kili did it 0 times, and it keeps the most fillers, backchannels and repetitions. Use it for formal
  speech too: turning a transcript into formal Tamil, a summary or a translation is a text model's job.
- **English** stays close to the original parakeet v3 it started from. It does not forget English.
- **Code-switching** is the default. English words come out in Latin script with the Tamil suffix attached
  (`meeting-க்கு`, `office-ல`): 89% of the English words in conversational Tamil (IndicVoices), 6% in Tamil script.
  English speech stays English: Tamil script appears in 0.1–0.2% of LibriSpeech and Common Voice outputs. On
  Indian-accented English (Svarah) it is 9.5% of clips but 1.5% of words, mostly places and names that have Tamil
  spellings (டெல்லி, லண்டன், ஜப்பான்), Indian-language words and one-word clips; the meaning is kept. (The loanword
  converter's recall of 0.64 is measured on the training labels, not on the model's output: converting the common
  loanwords consistently taught the model the rule, and it applies it to most of the loanwords the converter left alone.)

**CPU speed**: the int8 ONNX encoder runs about 28× faster than real time on 8 threads, with a peak of about 2.8 GB
of memory. No PyTorch is needed. We do not compare speed with the other models: our CPU path is tuned and theirs ran
through reference code, which would flatter us.¹

¹ Speed is mostly engineering. [fast-gpu-asr](https://github.com/SoundsGoodAI/fast-gpu-asr) runs the original
parakeet v3 at 6,482× real time on one A100 and 19,399× on a B300 (batch 256); kili has the same architecture.

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
uv run python -m kili.infer.transcribe --model shrirajh/parakeet-tamil-kili clip.wav --timestamps   # word start / end times
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
  warm-started from the mean of the original parakeet v3's rows for its romanised spelling (க்கு starts as "kku"), so after the surgery
  the model scores exactly like the original parakeet v3.
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

Built by shrirajh with Claude Opus 5.5 [1m], via [Claude Code](https://claude.com/claude-code), as a coding and research
assistant. Human in the loop throughout: a person directed every step and checked every result. Claude was not used
autonomously.
