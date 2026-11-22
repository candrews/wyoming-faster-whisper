"""Code for transcription using Qwen3-ASR ONNX exports.

Qwen3-ASR is a speech LLM: an audio encoder feeds projected audio embeddings into
a Qwen3 decoder that generates the transcript token by token. The ONNX export
splits it into several graphs, so unlike the other backends there is no library
that drives it for us - the greedy decode loop lives here.

The payoff is context biasing. Qwen3-ASR accepts free-form text in the chat
template's system turn to bias decoding toward specific spellings, so
``--initial-prompt`` can carry Home Assistant entity names ("Vocabulary: Ecobee,
office lamp.") and fix names that would otherwise be transcribed phonetically.

Three decoder layouts are supported, chosen by which files the model directory has:

*merged* (``decoder_merged.int4.onnx``) - one graph taking a KV cache *and* a
dynamic sequence length. Because the biasing prompt sits in the system turn,
ahead of the audio, its KV depends only on the prompt tokens, so it is computed
once and reused for every later utterance. That matters because the prompt is not
free: a 50-name list is ~180 tokens of prefill, which on a Pi 5 took a 3.2 s
command from 1.47 s to 3.42 s. Reusing it gives 2.20 s, and the graph also drops
the in-graph embedding table (1407 -> 785 MB on disk, 2.25 -> 1.55 GB peak RSS).

*merged + GQA* (``decoder_gqa.int4.onnx`` next to the merged files) - the same
graph and weights, with each layer's attention replaced by onnxruntime's fused
GroupQueryAttention, so the KV cache stays in one buffer per layer that each
pass appends to in place. Preferred on the CPU when present.

*split* (``decoder_init`` + ``decoder_step``) - the original export. ``decoder_init``
has no KV input and ``decoder_step`` is pinned to one token, so the prompt must be
re-prefilled every utterance. Kept as a fallback so existing model directories
keep working.

onnxruntime is imported at module scope (not lazily inside __init__) so that
importing this module raises ImportError when it is absent. ModelLoader relies on
that to detect the backend and fall back to faster-whisper, matching
onnx_asr_handler/funasr_handler.
"""

import json
import logging
import threading
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from .const import StreamingSession, Transcriber, qwen3_asr_language
from .device import disable_onnx_spinning, onnx_providers, warn_if_no_onnx_gpu
from .vad import StreamingSpeechClipper

_LOGGER = logging.getLogger(__name__)

_RATE = 16000

# Mel front-end (identical to Whisper's).
_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 128
_FMIN = 0.0
_FMAX = 8000.0

# Chat template token ids. NOTE: the reference prompt builder in
# andrewleech/qwen3-asr-onnx hardcodes 9125 and 882 here, which decode to
# " Current" and " time" rather than "system" and "user".
_IM_START = 151644
_IM_END = 151645
_SYSTEM = 8948
_USER = 872
_ASSISTANT = 77091
_NEWLINE = 198
_AUDIO_START = 151669
_AUDIO_END = 151670
_AUDIO_PAD = 151676
_ASR_TEXT = 151704  # <asr_text>: everything before it is the language preamble
_EOS_IDS = frozenset((151643, 151645))

_MAX_NEW_TOKENS = 256

# Most draft tokens checked in one decoder step after the decode has left the
# draft and found it again. A step is bound by reading the weights, so checking
# a few tokens costs about the same as generating one.
_MAX_GUESS_TOKENS = 16

# Only these files are needed for int4 inference; the repo also holds multi-GB
# FP32 weights and tarballs that we must not pull down.
_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "embed_tokens.bin",
    "encoder.int4.onnx",
    "encoder.int4.onnx.data",
    # Merged layout, and its GroupQueryAttention graph (same weights file).
    "decoder_merged.int4.onnx",
    "decoder_merged.int4.onnx.data",
    "decoder_gqa.int4.onnx",
    # Split layout. A repo has one layout or the other, and patterns that match
    # nothing are simply skipped, so one list serves both.
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",
]

# Presence of this file selects the merged layout.
_MERGED_DECODER = "decoder_merged.int4.onnx"

# The merged decoder with each layer's attention replaced by onnxruntime's
# GroupQueryAttention, reading decoder_merged's weights file. Its KV cache is one
# buffer per layer that the graph appends to in place, instead of a stacked
# cache passed in and returned whole on every step: that copy grew with the
# context, and cutting rejected draft tokens off it was one more. GQA also masks
# causally itself, so there is no attention mask to build. Used on the CPU only:
# onnxruntime's CUDA kernel for it takes fp16 alone, so that provider keeps the
# stacked graph.
_GQA_DECODER = "decoder_gqa.int4.onnx"

# GQA KV buffers hold at least this many positions (grown for longer prompts).
# np.zeros leaves untouched pages unallocated, so only positions actually used
# cost memory.
_KV_CAPACITY = 1024
# Idle GQA KV buffers kept for reuse. Two decodes can overlap: a streaming pass
# that is being aborted may still be running when the next one starts.
_KV_POOL_SIZE = 2

# Optional input of the merged decoder and of decoder_init (same name and
# meaning as in transformers): compute logits only for the last this-many
# positions. A prefill of ~100 positions reads one or a few rows, and the
# 151936-wide lm_head over the rest was 16-21% of prefill time on the CPU, plus
# ~60 MB copied to the host on Vulkan.
_NUM_LOGITS_TO_KEEP = "num_logits_to_keep"

# Output of the GQA decoder: ArgMax of logits over the vocabulary, int64
# [batch, num_logits]. Decoding is always greedy, so token ids are all that is
# needed, and they are what the binding returns instead of 151936-wide rows.
_NEXT_TOKENS = "next_tokens"

# Streaming (--qwen3-asr-streaming): decode all audio so far again every this
# many seconds of new audio, checking the previous pass's transcript as a draft.
_STREAM_CHUNK_SECONDS = 2.0
# With --vad-clip, the silence after speech that starts the final pass early.
# The clip's trailing pad (--vad-clip-pad-ms) must have arrived too, so this
# only matters for short pads: it keeps a brief pause from starting one.
_SPEECH_END_SILENCE_MS = 300

# Additive attention mask: 0 where attending is allowed, this where it is not.
# pylint cannot see finfo's attributes: numpy (2.5 here) ships no Python source for it.
_MASK_BLOCKED = np.finfo(np.float32).min  # pylint: disable=no-member


def _mel_filterbank() -> np.ndarray:
    """Slaney-normalized mel filterbank, matching librosa.filters.mel(norm="slaney")."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        mels = freq / f_sp
        above = freq >= min_log_hz
        mels[above] = min_log_mel + np.log(freq[above] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        freqs = mels * f_sp
        above = mels >= min_log_mel
        freqs[above] = min_log_hz * np.exp(logstep * (mels[above] - min_log_mel))
        return freqs

    fft_freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / _RATE)
    mel_points = np.linspace(
        hz_to_mel(np.array([_FMIN]))[0], hz_to_mel(np.array([_FMAX]))[0], _N_MELS + 2
    )
    band_hz = mel_to_hz(mel_points)

    diff = np.diff(band_hz)
    ramps = band_hz[:, np.newaxis] - fft_freqs[np.newaxis, :]
    weights = np.zeros((_N_MELS, len(fft_freqs)), dtype=np.float32)
    for i in range(_N_MELS):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: equal area per filter.
    weights *= (2.0 / (band_hz[2 : _N_MELS + 2] - band_hz[:_N_MELS]))[:, np.newaxis]
    return weights


def _log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper-compatible log-mel spectrogram, shape [1, n_mels, time]."""
    padded = np.pad(audio, _N_FFT // 2, mode="reflect")
    # Periodic Hann (np.hanning is symmetric).
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)

    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP_LENGTH]
    spec = np.fft.rfft(frames * window, n=_N_FFT, axis=-1)
    magnitudes = (np.abs(spec) ** 2).T.astype(np.float32)

    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # match WhisperFeatureExtractor's frame count
    return log_spec[np.newaxis, :, :].astype(np.float32)


def _bytes_to_samples(audio_bytes: bytes) -> np.ndarray:
    """Convert 16-bit mono PCM bytes to a float32 sample array."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _system_ids(context_ids: Sequence[int]) -> List[int]:
    """The chat template's system turn, which is where the biasing context goes.

    Split out because it is exactly the span whose KV the merged decoder caches:
    it precedes the audio, so nothing in it changes from one utterance to the next.
    """
    return [_IM_START, _SYSTEM, _NEWLINE, *context_ids, _IM_END, _NEWLINE]


def _accept_draft(predicted: np.ndarray, draft: Sequence[int]) -> List[int]:
    """Greedy tokens implied by one prefill over a draft, as far as it holds.

    ``predicted`` has the greedy token for the last prompt position and one per
    draft token: entry j is what greedy decoding would pick in place of
    ``draft[j]``. Draft tokens are accepted while they match that pick; the
    returned list ends with the model's own token at the first mismatch (or after
    the whole draft), which has not been fed through the model yet. A matched
    end-of-text stops it early.
    """
    tokens: List[int] = []
    for j, guess in enumerate(draft):
        token = int(predicted[j])
        tokens.append(token)
        if (token != guess) or (token in _EOS_IDS):
            return tokens

    tokens.append(int(predicted[len(draft)]))
    return tokens


def _draft_continuation(
    draft: Sequence[int], tokens: Sequence[int], cursor: int
) -> Tuple[List[int], int]:
    """Where the draft picks up again after the decode has left it.

    Finds the last two decoded tokens in the draft, ending at or after
    ``cursor``, and returns what follows them there along with that guess's
    index in the draft. A draft often differs from the result only locally
    ("thirteen" against "13"), and agrees again a word later. One token alone
    is too weak a match: punctuation and short words recur all over a draft,
    and every wrong guess costs a wider decoder step.
    """
    if len(tokens) < 2:
        return [], cursor

    tail = list(tokens[-2:])
    for i in range(max(cursor - 2, 0), len(draft) - 2):
        if list(draft[i : i + 2]) == tail:
            return list(draft[i + 2 : i + 2 + _MAX_GUESS_TOKENS]), i + 2
    return [], cursor


def _continue_greedy(
    predicted: np.ndarray,
    draft: Sequence[int],
    n_prompt: int,
    step: Callable[[List[int], int], np.ndarray],
    multi_token_steps: bool,
) -> List[int]:
    """Finish a greedy decode from a prefill that also ran ``draft``.

    ``predicted`` holds the prefill's greedy tokens from the last prompt position
    on, and the KV cache covers the prompt and the whole draft.
    ``step(tokens, position)`` runs the decoder over ``tokens`` on top of the
    first ``position`` entries of the cache and returns the greedy token after
    each. Any entries past ``position`` are rejected draft tokens: the step must
    not attend to them (the stacked layouts cut them off, the GQA one writes over
    them).

    With ``multi_token_steps`` (the merged decoder; the split one's step graph
    takes a single token), each step after a mismatch also checks where the
    draft resumes, so one differing word does not cost the rest of the draft.
    """
    tokens = _accept_draft(predicted, draft)
    accepted = len(tokens) - 1
    total_accepted = accepted

    position = n_prompt + accepted
    cursor = accepted
    while (tokens[-1] not in _EOS_IDS) and (len(tokens) < _MAX_NEW_TOKENS):
        guess: List[int] = []
        start = cursor
        if multi_token_steps and draft:
            guess, start = _draft_continuation(draft, tokens, cursor)

        new = _accept_draft(step([tokens[-1]] + guess, position), guess)
        accepted = len(new) - 1
        total_accepted += accepted
        tokens.extend(new)

        # The step fed tokens[-1] and the whole guess; keep only what held.
        if guess:
            cursor = start + accepted
        position += 1 + accepted

    if draft:
        _LOGGER.debug("Draft: %s of %s token(s) accepted", total_accepted, len(draft))

    return tokens[:_MAX_NEW_TOKENS]


def _truncated_kv(
    keys: np.ndarray, values: np.ndarray, length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """A stacked KV cache cut back to its first ``length`` positions."""
    if keys.shape[3] == length:
        return keys, values
    return (
        np.ascontiguousarray(keys[:, :, :, :length]),
        np.ascontiguousarray(values[:, :, :, :length]),
    )


def _takes_num_logits_to_keep(session: ort.InferenceSession) -> bool:
    """Whether a decoder graph can skip the logits of positions nobody reads.

    Older exports lack the input and return logits for every position. Newer
    ones give it a default of "all of them", so that older callers still work,
    which makes onnxruntime list it as an overridable initializer, not an input.
    """
    return _NUM_LOGITS_TO_KEEP in {
        arg.name
        for arg in session.get_inputs() + session.get_overridable_initializers()
    }


class _KvBuffer:
    """One decode's KV cache for the GQA decoder, bound in place.

    Each layer's key and value buffer is bound as both its past input and its
    present output, so a pass reads the cache where it is and writes only the
    new positions into it. Positions past the length a pass is told about are
    never read, which is all it takes to drop rejected draft tokens.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        shape: Tuple[int, int, int],
        capacity: int,
    ) -> None:
        layers, kv_heads, head_dim = shape
        self.keys = np.zeros(
            (layers, 1, kv_heads, capacity, head_dim), dtype=np.float32
        )
        self.values = np.zeros_like(self.keys)
        self.binding = session.io_binding()
        # Bound first, so it is get_outputs()[0] (re-binding keeps its place).
        self.binding.bind_output(_NEXT_TOKENS)
        for layer in range(layers):
            for kind, buffer in (("key", self.keys), ("value", self.values)):
                value = ort.OrtValue.ortvalue_from_numpy(buffer[layer])
                self.binding.bind_ortvalue_input(
                    f"past_key_values.{layer}.{kind}", value
                )
                self.binding.bind_ortvalue_output(f"present.{layer}.{kind}", value)

        # The system turn whose KV the first positions hold, if any.
        self.prefix: Optional[Tuple[int, ...]] = None

    @property
    def capacity(self) -> int:
        return self.keys.shape[3]


def _causal_mask(q_len: int, past_len: int) -> np.ndarray:
    """[1, 1, q_len, past_len + q_len] additive mask for the merged decoder.

    New tokens may attend to everything already in the cache, and causally among
    themselves. Built here rather than inside the graph because deriving it from
    input shapes is the kind of shape arithmetic the ONNX exporter freezes into
    constants.
    """
    new = np.triu(np.full((q_len, q_len), _MASK_BLOCKED, dtype=np.float32), k=1)
    if past_len == 0:
        return new[np.newaxis, np.newaxis]

    past = np.zeros((q_len, past_len), dtype=np.float32)
    return np.concatenate([past, new], axis=-1)[np.newaxis, np.newaxis]


class Qwen3AsrTranscriber(Transcriber):
    """Wrapper for a Qwen3-ASR ONNX export (encoder + split decoder)."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        cpu_threads: int = 4,
        device: str = "cpu",
    ) -> None:
        """Initialize model."""
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    model_id,
                    cache_dir=str(Path(cache_dir).resolve()),
                    local_files_only=local_files_only,
                    allow_patterns=_ALLOW_PATTERNS,
                )
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        disable_onnx_spinning(options)
        # The int4 weights are MatMulNBits nodes. The CUDA provider supports
        # them, but the quantization is where most of this model's speed comes
        # from, so the GPU win here is smaller than for the fp32 backends.
        session_args = {
            "sess_options": options,
            "providers": onnx_providers(device),
        }

        self._encoder = ort.InferenceSession(
            str(model_dir / "encoder.int4.onnx"), **session_args
        )

        # Ask the session that was actually created, not onnxruntime's advertised
        # provider list: a CUDA provider that fails to load is still advertised.
        warn_if_no_onnx_gpu(device, self._encoder.get_providers())

        self._gqa: Optional[ort.InferenceSession] = None
        self._merged: Optional[ort.InferenceSession] = None
        self._keeps_logits = False
        self._decoder_init: Optional[ort.InferenceSession] = None
        self._decoder_step: Optional[ort.InferenceSession] = None

        # Idle GQA KV buffers, and the (layers, kv_heads, head_dim) to make new
        # ones with.
        self._kv_lock = threading.Lock()
        self._kv_pool: List[_KvBuffer] = []
        self._kv_shape: Tuple[int, int, int] = (0, 0, 0)

        if (model_dir / _GQA_DECODER).is_file() and (
            self._encoder.get_providers() == ["CPUExecutionProvider"]
        ):
            self._gqa = ort.InferenceSession(
                str(model_dir / _GQA_DECODER), **session_args
            )
            self._keeps_logits = _takes_num_logits_to_keep(self._gqa)
            past = [arg for arg in self._gqa.get_inputs() if arg.name.endswith(".key")]
            # past_key_values.{i}.key is [batch, kv_heads, capacity, head_dim].
            self._kv_shape = (len(past), past[0].shape[1], past[0].shape[3])
            _LOGGER.debug("Using GQA decoder (KV cache updated in place)")
        elif (model_dir / _MERGED_DECODER).is_file():
            self._merged = ort.InferenceSession(
                str(model_dir / _MERGED_DECODER), **session_args
            )
            # present_keys is [num_layers, batch, kv_heads, seq, head_dim]; the
            # non-sequence dims are static, so the empty cache that starts a
            # prefill can be shaped from the graph instead of from config.json.
            kv_shape = self._merged.get_outputs()[1].shape
            self._empty_kv = np.zeros(
                (kv_shape[0], 1, kv_shape[2], 0, kv_shape[4]), dtype=np.float32
            )
            self._keeps_logits = _takes_num_logits_to_keep(self._merged)
            _LOGGER.debug("Using merged decoder (biasing prompt KV is reused)")
        else:
            self._decoder_init = ort.InferenceSession(
                str(model_dir / "decoder_init.int4.onnx"), **session_args
            )
            self._keeps_logits = _takes_num_logits_to_keep(self._decoder_init)
            self._decoder_step = ort.InferenceSession(
                str(model_dir / "decoder_step.int4.onnx"), **session_args
            )
            _LOGGER.debug(
                "Using split decoder; the biasing prompt is re-prefilled every "
                "utterance. A model directory with %s avoids that.",
                _MERGED_DECODER,
            )

        # Cached KV for the system turn, merged layout only. Exactly one entry:
        # Home Assistant sends the same name list every time, and each entry is
        # substantial (~42 MB for a 180-token prompt), so keeping a history would
        # cost far more memory than it could ever save.
        self._prefix_lock = threading.Lock()
        self._prefix_key: Optional[Tuple[int, ...]] = None
        self._prefix_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None

        with open(model_dir / "config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        # Keep the embedding table in fp16 and cast a single row per generated
        # token. Casting the whole table up front costs ~300MB of RSS for nothing.
        hidden_size = config["decoder"]["hidden_size"]
        self._embed_tokens = np.fromfile(
            model_dir / "embed_tokens.bin", dtype=np.float16
        ).reshape(-1, hidden_size)

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._mel_filters = _mel_filterbank()
        self._encoder_inputs = [inp.name for inp in self._encoder.get_inputs()]
        self._prompt_cache: Dict[Tuple[str, Optional[str]], List[int]] = {}

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def _context_ids(
        self, initial_prompt: Optional[str], language: Optional[str]
    ) -> Tuple[List[int], List[int]]:
        """Tokenize the biasing context and the forced-language suffix."""
        key = (initial_prompt or "", language)
        cached = self._prompt_cache.get(key)
        if cached is None:
            context = (
                self._tokenizer.encode(initial_prompt, add_special_tokens=False).ids
                if initial_prompt
                else []
            )
            language_name = qwen3_asr_language(language)
            if language_name:
                suffix = self._tokenizer.encode(
                    f"language {language_name}", add_special_tokens=False
                ).ids + [_ASR_TEXT]
            else:
                # Auto-detect: the answer always opens "language <Name><asr_text>",
                # so start it with "language" and leave the model only the name.
                # Left to choose the first token itself, it can tip into
                # something else on a near-tie, skip the preamble and derail the
                # whole transcript ("Start a timer for 25 minutes" came out as
                # "ology" on onnxruntime 1.22 on a CPU).
                suffix = self._tokenizer.encode(
                    "language", add_special_tokens=False
                ).ids
            # Cache as one list so a single dict lookup covers both.
            cached = [len(context)] + context + suffix
            self._prompt_cache[key] = cached

        n_context = cached[0]
        return cached[1 : 1 + n_context], cached[1 + n_context :]

    def _build_prompt_ids(
        self,
        n_audio_tokens: int,
        context_ids: Sequence[int],
        language_ids: Sequence[int],
    ) -> List[int]:
        """Build the chat-template prompt around the audio placeholders.

        <|im_start|>system\n{context}<|im_end|>\n
        <|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n
        <|im_start|>assistant\n{language {Name}<asr_text>}
        """
        return (
            _system_ids(context_ids)
            + [_IM_START, _USER, _NEWLINE, _AUDIO_START]
            + [_AUDIO_PAD] * n_audio_tokens
            + [_AUDIO_END, _IM_END, _NEWLINE]
            + [_IM_START, _ASSISTANT, _NEWLINE, *language_ids]
        )

    def _run_merged(
        self,
        input_embeds: np.ndarray,
        positions: Union[Sequence[int], np.ndarray],
        past_keys: np.ndarray,
        past_values: np.ndarray,
        num_logits: int,
        run_options: Optional[ort.RunOptions] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One pass of the merged decoder over ``input_embeds``.

        Returns the greedy token for each of the last ``num_logits`` positions
        only, and the new KV cache. With an export that takes num_logits_to_keep,
        the logits of the other positions are never computed; with an older one
        they are computed and dropped here.
        """
        assert self._merged is not None
        assert num_logits >= 1  # 0 would mean "every position" to the graph
        feeds = {
            "input_embeds": input_embeds,
            "position_ids": np.asarray(positions, dtype=np.int64)[np.newaxis, :],
            "attention_mask": _causal_mask(input_embeds.shape[1], past_keys.shape[3]),
            "past_keys": past_keys,
            "past_values": past_values,
        }
        if self._keeps_logits:
            feeds[_NUM_LOGITS_TO_KEEP] = np.array([num_logits], dtype=np.int64)

        logits, keys, values = self._merged.run(
            ["logits", "present_keys", "present_values"], feeds, run_options
        )
        return np.argmax(logits[0, -num_logits:], axis=-1), keys, values

    def _prefix_cache(
        self, system_ids: List[int], run_options: Optional[ort.RunOptions] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """KV for the system turn, computed once and reused while it is unchanged."""
        key = tuple(system_ids)
        with self._prefix_lock:
            if (self._prefix_key == key) and (self._prefix_kv is not None):
                return self._prefix_kv

        # Run outside the lock: this is a model pass, and holding the lock would
        # serialize concurrent requests behind it. Two callers racing here just
        # compute the same thing twice and store the same result.
        _, keys, values = self._run_merged(
            self._embed(system_ids),
            np.arange(len(system_ids)),
            self._empty_kv,
            self._empty_kv,
            1,  # only the KV is wanted; the graph has no way to ask for none
            run_options,
        )

        with self._prefix_lock:
            self._prefix_key = key
            self._prefix_kv = (keys, values)

        _LOGGER.debug("Cached KV for a %s-token system turn", len(system_ids))
        return keys, values

    def _generate_merged(
        self,
        audio_features: np.ndarray,
        prompt_ids: List[int],
        n_system: int,
        draft: Sequence[int],
        run_options: Optional[ort.RunOptions] = None,
    ) -> List[int]:
        """Greedy decode reusing the system turn's KV across utterances."""
        past_keys, past_values = self._prefix_cache(prompt_ids[:n_system], run_options)

        predicted, past_keys, past_values = self._run_merged(
            self._embed_after_system(audio_features, prompt_ids, n_system, draft),
            np.arange(n_system, len(prompt_ids) + len(draft)),
            past_keys,
            past_values,
            len(draft) + 1,  # the last prompt position and each draft token
            run_options,
        )

        cache = [past_keys, past_values]

        def step(tokens: List[int], position: int) -> np.ndarray:
            step_predicted, cache[0], cache[1] = self._run_merged(
                self._embed(tokens),
                np.arange(position, position + len(tokens)),
                *_truncated_kv(cache[0], cache[1], position),
                len(tokens),
                run_options,
            )
            return step_predicted

        return _continue_greedy(
            predicted, draft, len(prompt_ids), step, multi_token_steps=True
        )

    def _acquire_kv(self, needed: int) -> _KvBuffer:
        """A GQA KV buffer with room for ``needed`` positions."""
        with self._kv_lock:
            buffer = self._kv_pool.pop() if self._kv_pool else None

        if (buffer is None) or (buffer.capacity < needed):
            assert self._gqa is not None
            capacity = max(_KV_CAPACITY, needed)
            buffer = _KvBuffer(self._gqa, self._kv_shape, capacity)

        return buffer

    def _release_kv(self, buffer: _KvBuffer) -> None:
        with self._kv_lock:
            if len(self._kv_pool) < _KV_POOL_SIZE:
                self._kv_pool.append(buffer)

    def _run_gqa(
        self,
        buffer: _KvBuffer,
        input_embeds: np.ndarray,
        past_len: int,
        num_logits: int,
        run_options: Optional[ort.RunOptions] = None,
    ) -> np.ndarray:
        """One pass of the GQA decoder over ``input_embeds``, appended to
        ``buffer`` at ``past_len``; the greedy token for each of the last
        ``num_logits`` positions."""
        assert self._gqa is not None
        assert num_logits >= 1  # 0 would mean "every position" to the graph
        total = past_len + input_embeds.shape[1]
        assert total <= buffer.capacity
        # Keep every fed array referenced until the run is over: the binding
        # can point at their memory rather than copy them.
        feeds = {
            "input_embeds": input_embeds,
            "position_ids": np.arange(past_len, total, dtype=np.int64)[np.newaxis],
            "seqlens_k": np.array([total - 1], dtype=np.int32),
            "total_sequence_length": np.array(total, dtype=np.int32),
        }
        if self._keeps_logits:
            feeds[_NUM_LOGITS_TO_KEEP] = np.array([num_logits], dtype=np.int64)
        for name, value in feeds.items():
            buffer.binding.bind_cpu_input(name, value)
        # Re-bound every run: a bound output keeps the shape of the first run
        # that filled it, and the number of rows changes from pass to pass.
        buffer.binding.bind_output(_NEXT_TOKENS)

        self._gqa.run_with_iobinding(buffer.binding, run_options)
        return buffer.binding.get_outputs()[0].numpy()[0, -num_logits:]

    def _load_prefix(
        self,
        buffer: _KvBuffer,
        system_ids: List[int],
        run_options: Optional[ort.RunOptions] = None,
    ) -> None:
        """Put the system turn's KV in the first positions of ``buffer``.

        Nothing to do when the buffer already holds it, which is the usual case:
        decodes never write there, and the prompt seldom changes.
        """
        key = tuple(system_ids)
        if buffer.prefix == key:
            return

        n_system = len(system_ids)
        buffer.prefix = None
        with self._prefix_lock:
            cached = self._prefix_kv if self._prefix_key == key else None

        if cached is not None:
            buffer.keys[:, :, :, :n_system] = cached[0]
            buffer.values[:, :, :, :n_system] = cached[1]
        else:
            self._run_gqa(
                buffer,
                self._embed(system_ids),
                0,
                1,  # only the KV is wanted; the graph has no way to ask for none
                run_options,
            )
            # Kept for the other pooled buffer, or a bigger one.
            cached = (
                buffer.keys[:, :, :, :n_system].copy(),
                buffer.values[:, :, :, :n_system].copy(),
            )
            with self._prefix_lock:
                self._prefix_key = key
                self._prefix_kv = cached
            _LOGGER.debug("Cached KV for a %s-token system turn", n_system)

        buffer.prefix = key

    def _generate_gqa(
        self,
        audio_features: np.ndarray,
        prompt_ids: List[int],
        n_system: int,
        draft: Sequence[int],
        run_options: Optional[ort.RunOptions] = None,
    ) -> List[int]:
        """Greedy decode with the GQA decoder; see _generate_merged."""
        input_embeds = self._embed_after_system(
            audio_features, prompt_ids, n_system, draft
        )

        # Every position a decode can reach: the prompt and draft, then at most
        # _MAX_NEW_TOKENS confirmed tokens plus one step's guess on top.
        needed = len(prompt_ids) + len(draft) + _MAX_NEW_TOKENS + _MAX_GUESS_TOKENS
        buffer = self._acquire_kv(needed)
        try:
            self._load_prefix(buffer, prompt_ids[:n_system], run_options)
            predicted = self._run_gqa(
                buffer, input_embeds, n_system, len(draft) + 1, run_options
            )

            def step(tokens: List[int], position: int) -> np.ndarray:
                return self._run_gqa(
                    buffer,
                    self._embed(tokens),
                    position,
                    len(tokens),
                    run_options,
                )

            return _continue_greedy(
                predicted, draft, len(prompt_ids), step, multi_token_steps=True
            )
        finally:
            self._release_kv(buffer)

    def _generate(
        self,
        audio_features: np.ndarray,
        prompt_ids: List[int],
        draft: Sequence[int],
        run_options: Optional[ort.RunOptions] = None,
    ) -> List[int]:
        """Greedy decode on the split layout: prefill, then one token at a time."""
        assert (self._decoder_init is not None) and (self._decoder_step is not None)
        decoder_step = self._decoder_step
        input_ids = prompt_ids + list(draft)
        # The last prompt position and each draft token.
        num_logits = len(draft) + 1

        feeds = {
            "input_ids": np.array(input_ids, dtype=np.int64)[np.newaxis, :],
            "position_ids": np.arange(len(input_ids), dtype=np.int64)[np.newaxis, :],
            "audio_features": audio_features,
            "audio_offset": np.array([prompt_ids.index(_AUDIO_PAD)], dtype=np.int64),
        }
        if self._keeps_logits:
            feeds[_NUM_LOGITS_TO_KEEP] = np.array([num_logits], dtype=np.int64)

        logits, past_keys, past_values = self._decoder_init.run(
            ["logits", "present_keys", "present_values"], feeds, run_options
        )

        cache = [past_keys, past_values]

        def step(tokens: List[int], position: int) -> np.ndarray:
            (token,) = tokens  # decoder_step is pinned to one token
            keys, values = _truncated_kv(cache[0], cache[1], position)
            step_logits, cache[0], cache[1] = decoder_step.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": self._embed([token]),
                    "position_ids": np.array([[position]], dtype=np.int64),
                    "past_keys": keys,
                    "past_values": values,
                },
                run_options,
            )
            return np.argmax(step_logits[0], axis=-1)

        return _continue_greedy(
            np.argmax(logits[0, -num_logits:], axis=-1),
            draft,
            len(prompt_ids),
            step,
            multi_token_steps=False,
        )

    def decode_tokens(
        self,
        audio: np.ndarray,
        language: Optional[str],
        initial_prompt: Optional[str] = None,
        draft: Sequence[int] = (),
        run_options: Optional[ort.RunOptions] = None,
    ) -> List[int]:
        """Greedy-decode float32 16kHz ``audio`` into the model's raw tokens.

        The result is the whole assistant turn, detected-language preamble too.
        ``draft`` is a guess at it (such as an earlier decode of most of the same
        audio). It does not change the result, only how fast it arrives: the
        draft is checked in the same single pass that reads the prompt, every
        token of it that matches greedy decoding is kept without a decoder step
        of its own, and decoding carries on from the first mismatch. The split
        decoder layout ignores it (see below).

        Setting ``run_options.terminate`` from another thread aborts the decode
        with an exception at the next graph node.
        """
        mel = _log_mel_spectrogram(audio, self._mel_filters)

        encoder_inputs = {self._encoder_inputs[0]: mel}
        if len(self._encoder_inputs) > 1:
            encoder_inputs[self._encoder_inputs[1]] = np.array(
                [mel.shape[2]], dtype=np.int64
            )
        audio_features = self._encoder.run(None, encoder_inputs, run_options)[0]

        context_ids, language_ids = self._context_ids(initial_prompt, language)
        prompt_ids = self._build_prompt_ids(
            audio_features.shape[1], context_ids, language_ids
        )
        draft = list(draft)[:_MAX_NEW_TOKENS]

        if (self._gqa is not None) or (self._merged is not None):
            generate = (
                self._generate_gqa if self._gqa is not None else self._generate_merged
            )
            return generate(
                audio_features,
                prompt_ids,
                len(_system_ids(context_ids)),
                draft,
                run_options,
            )

        # No draft on the split layout. It would be checked in decoder_init but
        # generated in decoder_step, two graphs that round differently, so there
        # a draft tipped near-ties far more often than on the merged layout: 13
        # of 260 test decodes, one of them with a draft identical to the
        # result. The split layout also gets none of the multi-token steps that
        # make drafts pay off.
        return self._generate(audio_features, prompt_ids, (), run_options)

    def tokens_to_text(self, tokens: Sequence[int]) -> str:
        """Turn the assistant turn into the transcript."""
        tokens = list(tokens)

        # The model writes the detected language before <asr_text>; drop it.
        if _ASR_TEXT in tokens:
            tokens = tokens[tokens.index(_ASR_TEXT) + 1 :]

        return self._tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _embed(self, token_ids: Sequence[int]) -> np.ndarray:
        """[1, len(token_ids), hidden] float32 input embeddings."""
        return self._embed_tokens[list(token_ids)].astype(np.float32)[np.newaxis]

    def _embed_after_system(
        self,
        audio_features: np.ndarray,
        prompt_ids: List[int],
        n_system: int,
        draft: Sequence[int],
    ) -> np.ndarray:
        """Embeddings of everything after the system turn: the audio
        placeholders (replaced by the audio features), the assistant preamble,
        and the draft."""
        input_embeds = self._embed(prompt_ids[n_system:] + list(draft))
        audio_at = prompt_ids.index(_AUDIO_PAD) - n_system
        input_embeds[0, audio_at : audio_at + audio_features.shape[1]] = audio_features[
            0
        ]
        return input_embeds

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio. Decoding is greedy, so
        beam_size is ignored. initial_prompt is used as biasing context rather
        than as a transcript prefix.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        return self.tokens_to_text(
            self.decode_tokens(_bytes_to_samples(audio_bytes), language, initial_prompt)
        )


class Qwen3AsrStreamingTranscriber(Qwen3AsrTranscriber):
    """Qwen3-ASR that does most of its work while the audio is still arriving.

    The model is not natively streaming: the encoder attends across its whole
    window and the mel front-end normalizes to the loudest frame, so the encoding
    of early audio changes as more arrives and nothing can be committed for good.
    Instead, every ``chunk_seconds`` of new audio, all of the audio so far is
    decoded again, with the previous pass's transcript as a draft (see
    decode_tokens). Checking a draft token costs a fraction of generating one,
    and a pass that heard most of the same audio is usually right about most of
    it, so by audio-stop the final pass is mostly checking rather than
    generating. A wrong draft costs only speed: the result is always the
    model's own greedy decode.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        cpu_threads: int = 4,
        device: str = "cpu",
        chunk_seconds: float = _STREAM_CHUNK_SECONDS,
        vad_clip: bool = False,
        vad_clip_threshold: float = 0.5,
        vad_clip_pad_ms: int = 400,
    ) -> None:
        super().__init__(
            model_id,
            cache_dir,
            local_files_only=local_files_only,
            cpu_threads=cpu_threads,
            device=device,
        )
        self.chunk_seconds = chunk_seconds
        self.vad_clip = vad_clip
        self.vad_clip_threshold = vad_clip_threshold
        self.vad_clip_pad_ms = vad_clip_pad_ms

    @property
    def supports_streaming(self) -> bool:
        return True

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> StreamingSession:
        clipper = (
            StreamingSpeechClipper(self.vad_clip_threshold, self.vad_clip_pad_ms)
            if self.vad_clip
            else None
        )
        return Qwen3AsrStreamingSession(
            self,
            language,
            initial_prompt,
            chunk_seconds=self.chunk_seconds,
            clipper=clipper,
        )


class Qwen3AsrStreamingSession(StreamingSession):
    """A single in-progress Qwen3-ASR transcription.

    accept_chunk() only buffers: the intermediate passes run on a background
    thread, so the caller is never held up by model work (which would also delay
    --vad-endpointing). Passes never queue up behind each other: if one is still
    running when the next is due, the next waits for it to finish, so a slow CPU
    simply gets fewer, larger passes.

    With --vad-clip, the final pass can start before the audio stops. Once the
    speaker has been quiet for the clip's trailing pad, the clipped audio is
    exactly what finish() will decode unless they speak again, so it is decoded
    straight away (taking priority over any intermediate pass), and finish()
    returns that result, or waits for it if it is still running. The client
    usually waits a moment of silence of its own before it stops the audio, so
    the transcript can be ready by then. If the speaker carries on after all,
    the early result is just the draft for the next pass.
    """

    def __init__(
        self,
        model: Qwen3AsrTranscriber,
        language: Optional[str],
        initial_prompt: Optional[str],
        chunk_seconds: float = _STREAM_CHUNK_SECONDS,
        clipper: Optional[StreamingSpeechClipper] = None,
    ) -> None:
        self.model = model
        # --vad-clip: silence around the speech is clipped off the final pass,
        # whose encoder and prefill cost both grow with the audio's length.
        self.clipper = clipper
        self.language = language
        self.initial_prompt = initial_prompt
        # Two bytes per 16-bit sample.
        self._chunk_bytes = max(2, int(chunk_seconds * _RATE) * 2)

        self._audio = bytearray()
        self._leftover = b""
        self._decoded_bytes = 0  # audio covered by the last pass started
        # The audio and tokens of the last pass to finish.
        self._result: Tuple[bytes, List[int]] = (b"", [])
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._worker_audio = b""  # what the worker is decoding
        self._run_options: Optional[ort.RunOptions] = None
        # --vad-clip: the clipped span the last speech-end pass was started on.
        self._speech_end: Optional[Tuple[int, int]] = None

    def accept_chunk(self, audio_bytes: bytes) -> None:
        audio_bytes = self._leftover + audio_bytes
        if len(audio_bytes) % 2:
            self._leftover = audio_bytes[-1:]
            audio_bytes = audio_bytes[:-1]
        else:
            self._leftover = b""

        self._audio.extend(audio_bytes)
        if self.clipper is not None:
            self.clipper.process(audio_bytes)
            span = self.clipper.settled_span(_SPEECH_END_SILENCE_MS)
            if span is not None:
                if span != self._speech_end:
                    self._speech_end = span
                    self._abort_worker(wait=False)
                    self._start_pass(bytes(self._audio[span[0] : span[1]]), True)

                # Nothing new to hear until the speaker carries on.
                return

        if (len(self._audio) - self._decoded_bytes) < self._chunk_bytes:
            return

        if (self._worker is not None) and self._worker.is_alive():
            return

        self._decoded_bytes = len(self._audio)
        self._start_pass(bytes(self._audio), False)

    def _start_pass(self, audio_bytes: bytes, final: bool) -> None:
        self._run_options = ort.RunOptions()
        self._worker_audio = audio_bytes
        self._worker = threading.Thread(
            target=self._run_pass,
            args=(audio_bytes, self._run_options, final),
            name="qwen3-asr-stream",
            daemon=True,
        )
        self._worker.start()

    def _abort_worker(self, wait: bool) -> None:
        if self._worker is None:
            return

        assert self._run_options is not None
        self._run_options.terminate = True
        if wait:
            self._worker.join()

        self._worker = None
        self._worker_audio = b""

    def _draft(self) -> List[int]:
        """The last finished pass's tokens, as a draft for the next one."""
        with self._lock:
            return list(self._result[1])

    def _run_pass(
        self, audio_bytes: bytes, run_options: "ort.RunOptions", final: bool
    ) -> None:
        """One pass on a background thread.

        An intermediate pass reads all audio so far; a ``final`` one, the
        settled --vad-clip span (only the log tells them apart: finish() reuses
        any pass that read exactly its audio).
        """
        try:
            tokens = self.model.decode_tokens(
                _bytes_to_samples(audio_bytes),
                self.language,
                self.initial_prompt,
                draft=self._draft(),
                run_options=run_options,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # Only a draft for the final pass is lost; it re-reads all audio.
            if run_options.terminate:
                _LOGGER.debug("Intermediate Qwen3-ASR pass aborted")
            else:
                _LOGGER.exception("Intermediate Qwen3-ASR pass failed")
            return

        with self._lock:
            self._result = (audio_bytes, tokens)

        _LOGGER.debug(
            "%s transcript (%.2fs): %s",
            "Speech-end" if final else "Partial",
            len(audio_bytes) / (2 * _RATE),
            self.model.tokens_to_text(tokens),
        )

    def finish(self) -> str:
        if self._leftover:
            self._audio.extend(self._leftover + b"\x00")
            self._leftover = b""

        if not self._audio:
            return ""

        audio = bytes(self._audio)
        if self.clipper is not None:
            clipped = self.clipper.clip(audio)
            if clipped:
                _LOGGER.debug(
                    "VAD clip: %d ms -> %d ms",
                    len(audio) * 1000 // (2 * _RATE),
                    len(clipped) * 1000 // (2 * _RATE),
                )
                audio = clipped
            else:
                _LOGGER.debug("VAD clip: no speech detected, using full audio")

        if (self._worker is not None) and (self._worker_audio == audio):
            # A pass on exactly this audio (with --vad-clip, normally the one
            # started at speech end) is the final pass started early.
            self._worker.join()
        else:
            # Abort rather than wait: a pass still running is behind the audio
            # the final pass is about to read anyway, and waiting for it would
            # add its remaining time to the latency this is meant to cut. Its
            # loss costs only the draft it would have improved.
            self._abort_worker(wait=True)

        with self._lock:
            result_audio, result_tokens = self._result

        if result_audio == audio:
            _LOGGER.debug("Final pass already ran before audio-stop")
            return self.model.tokens_to_text(result_tokens)

        tokens = self.model.decode_tokens(
            _bytes_to_samples(audio),
            self.language,
            self.initial_prompt,
            draft=self._draft(),
        )
        return self.model.tokens_to_text(tokens)
