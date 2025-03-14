# Standard library imports
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

# Third-party imports
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from scipy import signal
# version of transformers should be <= 4.48.0 for at least Phi3
# source: https://github.com/huggingface/transformers/issues/36071
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# pip install --upgrade huggingface_hub
# huggingface-cli login
# or
# from huggingface_hub import login
# login()
# from notebooks
# from huggingface_hub import notebook_login
# notebook_login()

# global_device = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = global_device
# device = torch.device(f'cuda:{global_device}')

# -----------------------------------------------------------------------------
# BaseLLM definition
# -----------------------------------------------------------------------------
class BaseLLM(ABC):
    """
    Abstract base class for AI model management.
    Handles basic model operations with minimal complexity.
    """
    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_loaded = False
        self.device = f"cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)

    @property
    def model(self):
        if not self._is_loaded:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        if not self._is_loaded:
            self.load()
        return self._tokenizer

    @property
    def processor(self):
        if not self._is_loaded:
            self.load()
        return self._processor

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError("Subclasses must implement load()")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._is_loaded = False


# -----------------------------------------------------------------------------
# WhisperASR definition
# -----------------------------------------------------------------------------
class WhisperASR(BaseLLM):
    """
    Whisper ASR implementation using HuggingFace Transformers.
    Accepts audio as either:
      - A tuple: (sample_rate, numpy_array)
      - A plain numpy array (assumed to be at 16 kHz)
    Always normalizes the audio into mono 16 kHz.
    """
    default_models = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
    }

    def __init__(
        self,
        model_size: str = "tiny",
        language: Optional[str] = None,
        task: str = "transcribe",
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        if model_size not in self.default_models:
            raise ValueError(f"Model size must be one of {list(self.default_models.keys())}")
        self.model_id = self.default_models[model_size]
        self.language = language
        self.task = task
        self.cache_dir = cache_dir
        self.target_sample_rate = 16000  # Always resample to 16 kHz

    def load(self) -> None:
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            ).to(self.device)
            self._is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def _resample(self, data: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        elif np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        if sample_rate is not None and sample_rate != self.target_sample_rate:
            new_length = int(round(len(data) * self.target_sample_rate / sample_rate))
            data = signal.resample(data, new_length)
        return data

    def transcribe(
        self,
        audio_data: Any,  # Either a numpy array or a tuple (sample_rate, numpy array)
        **decoding_options: Any
    ) -> str:
        if audio_data is None:
            return "No audio data provided."
        if isinstance(audio_data, tuple):
            sample_rate, data = audio_data
        else:
            data = audio_data
            sample_rate = self.target_sample_rate
        data = self._resample(data=data, sample_rate=sample_rate)
        inputs = self.processor(
            data,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)
        generation_kwargs = {
            "task": self.task,
            "language": self.language,
            **decoding_options
        }
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()


   
# -----------------------------------------------------------------------------
# ChatLLM definition
# -----------------------------------------------------------------------------
class ChatLLM(BaseLLM):
    """
    Chat-based Language Model implementation using HuggingFace Transformers.
    """
    AVAILABLE_MODELS: Dict[str, str] = {
        "smol": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "exaone": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "granite": "ibm-granite/granite-3.1-2b-instruct",
        "phi": "microsoft/Phi-3-mini-4k-instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "gemma": "google/gemma-2-9b-it",
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct"
    }

    def __init__(
        self,
        model_name: str = "smol",
        cache_dir: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        conv_history_limit: int = 5,
    ) -> None:
        super().__init__()
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model_name '{model_name}'. Must be one of: {list(self.AVAILABLE_MODELS.keys())}"
            )
        self.model_name: str = model_name
        self.model_id: str = self.AVAILABLE_MODELS[model_name]
        self.cache_dir: Optional[str] = cache_dir
        self.max_length: int = max_length
        self.temperature: float = temperature
        self.do_sample: bool = do_sample
        self.conv_history: List[Dict[str, str]] = []
        self.conv_history_limit = conv_history_limit
        self.logger = logging.getLogger(__name__)

    def load(self) -> None:
        self.logger.info(f"Loading model {self.model_id} on {self.device}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
            self._processor = self._tokenizer
            self._is_loaded = True
            self.logger.info(f"Successfully loaded {self.model_name} model on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load LLM model '{self.model_id}': {e}")

    def _prepare_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.conv_history:
            messages.extend(self.conv_history)
        messages.append({"role": "user", "content": prompt})
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except (AttributeError, NotImplementedError):
            self.logger.warning(f"Chat template not available for {self.model_name}, using fallback formatting")
            formatted = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            formatted += "\nassistant: "
        return formatted

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        remember_conversation: bool = True,
        **generate_kwargs: Any
    ) -> str:
        formatted_prompt = self._prepare_prompt(prompt, system_prompt)
        try:
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            input_token_length = inputs.input_ids.shape[1] # len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, stream_output=stdout) if stream else None
            max_tokens = max_new_tokens if max_new_tokens is not None else (self.max_length - inputs.input_ids.shape[1])
            do_sample = do_sample if do_sample is not None else self.do_sample
            temperature = temperature if temperature is not None and do_sample else (self.temperature if do_sample else 1.0)
            generation_config = {
                # "streamer": streamer,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                **generate_kwargs
            }
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                )
            # Remove the prompt tokens using the tokenized input length
            new_tokens = outputs[0][input_token_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if remember_conversation:
                self.conv_history.append({"role": "user", "content": prompt})
                self.conv_history.append({"role": "assistant", "content": response})
                # Enforce the conversation history limit (each turn consists of a pair of messages)
                max_entries = self.conv_history_limit * 2  # user and assistant messages
                if len(self.conv_history) > max_entries:
                    self.conv_history = self.conv_history[-max_entries:]
            return response
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        remember_conversation: bool = True,
        **generate_kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Generate text in a streaming manner using TextIteratorStreamer.
        """
        formatted_prompt = self._prepare_prompt(prompt, system_prompt)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_token_length = inputs.input_ids.shape[1]
        max_tokens = max_new_tokens if max_new_tokens is not None else (self.max_length - input_token_length)
        do_sample = do_sample if do_sample is not None else self.do_sample
        temperature = temperature if temperature is not None and do_sample else (self.temperature if do_sample else 1.0)

        # Create the TextIteratorStreamer.
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_config = {
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **generate_kwargs
        }

        # run the generation in a separate thread.
        thread = threading.Thread(target=lambda: self.model.generate(**inputs, **generation_config))
        thread.start()

        generated_text = ""
        # stream tokens as they become available.
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

        thread.join()
        
        # If we want to remember the conversation, append both user prompt and final response
        if remember_conversation:
            self.conv_history.append({"role": "user", "content": prompt})
            final_response = generated_text.strip()
            self.conv_history.append({"role": "assistant", "content": final_response})
            # enforce the conversation history limit
            max_entries = self.conv_history_limit * 2  # user+assistant pairs
            if len(self.conv_history) > max_entries:
                self.conv_history = self.conv_history[-max_entries:]
                
        # finally, not sure if need to yield the complete response again
        # yield generated_text
    
    def save_conversation(self, file_path: str) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conv_history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Conversation saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {str(e)}")
            raise IOError(f"Failed to save conversation: {e}")

    def load_conversation(self, file_path: str) -> None:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conv_history = json.load(f)
            self.logger.info(f"Conversation loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load conversation: {str(e)}")
            raise IOError(f"Failed to load conversation: {e}")
        

# -----------------------------------------------------------------------------
# Default settings for XTTS-v2
# -----------------------------------------------------------------------------
default_xtts_settings = {
    "samplerate": 24000,
    "temperature": 0.75,
    "length_penalty": 1.0,
    "num_beams": 1,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
    "enable_text_splitting": False,
    "use_deepspeed": False,
}

# -----------------------------------------------------------------------------
# XTTSv2 Class Implementation
# -----------------------------------------------------------------------------
class XTTSv2(BaseLLM):
    """
    XTTSv2 model management class for Coqui XTTS-v2.

    Features:
      - Downloads and lazy-loads the model.
      - Performs inference with configurable generation parameters.
      - Supports both full and streaming inference.
      - Handles generated audio output: can return a tensor for in-app playback or save to a file.
    
    Designed for seamless integration into a Gradio UI.
    """
    def __init__(self, hf_repo: str = 'coqui/XTTS-v2', cache_dir: str = "./cache"):
        super().__init__()
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.downloaded = False
        self.lock = threading.Lock()

    def download_model(self) -> None:
        """
        Download the model files from Hugging Face Hub if not already available.
        """
        if not self.downloaded:
            self.logger.info("Downloading XTTS-v2 model files from Hugging Face Hub...")
            # snapshot_download(repo_id=self.hf_repo, local_dir=self.cache_dir)
            self.config_path = hf_hub_download(repo_id=self.hf_repo, filename="config.json", local_dir=self.cache_dir)
            self.model_path = hf_hub_download(repo_id=self.hf_repo, filename="model.pth", local_dir=self.cache_dir)
            self.vocab_path = hf_hub_download(repo_id=self.hf_repo, filename="vocab.json", local_dir=self.cache_dir)
            self.downloaded = True
            self.logger.info("Model files downloaded successfully.")
        else:
            self.logger.info("Model files already downloaded.")

    def load(self) -> None:
        """
        Load the XTTS-v2 model checkpoint using the downloaded files.
        """
        self.logger.info("Loading XTTS-v2 model...")
        if not self.downloaded:
            self.download_model()
        try:
            config = XttsConfig()
            config.models_dir = os.path.join("models", "tts")
            config.load_json(self.config_path)

            tts = Xtts.init_from_config(config)

            with self.lock:
                tts.load_checkpoint(
                    config,
                    checkpoint_path=self.model_path,
                    vocab_path=self.vocab_path,
                    use_deepspeed=default_xtts_settings['use_deepspeed'],
                    eval=True
                )
            if 'cuda' in self.device:
                tts.cuda()

            self._model = tts
            self._is_loaded = True
            self.logger.info("XTTS-v2 model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading XTTS-v2 model: {e}")
            self._is_loaded = False
            raise e

    def _build_gen_kwargs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default generation parameters with any provided overrides.

        :param overrides: Generation parameter overrides (e.g., temperature, speed).
        :return: A dictionary of updated generation parameters.
        """
        defaults = {
            "temperature": default_xtts_settings["temperature"],
            "length_penalty": default_xtts_settings["length_penalty"],
            "num_beams": default_xtts_settings["num_beams"],
            "repetition_penalty": default_xtts_settings["repetition_penalty"],
            "top_k": default_xtts_settings["top_k"],
            "top_p": default_xtts_settings["top_p"],
            "speed": default_xtts_settings["speed"],
            "enable_text_splitting": default_xtts_settings["enable_text_splitting"],
        }
        defaults.update(overrides)
        return defaults

    def inference(self, text: str, language: str, gpt_cond_latent, speaker_embedding, save_path: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """
        Generate speech audio from input text.

        :param text: The text to synthesize.
        :param language: The language code (e.g. "en", "es").
        :param gpt_cond_latent: Conditioning latent computed from reference audio.
        :param speaker_embedding: Speaker embedding for voice cloning.
        :param save_path: Optional file path to save the generated audio as a WAV file.
        :param overrides: Generation parameter overrides (e.g., temperature, speed).
        :return: A dictionary with the waveform tensor, samplerate, and, if saved, the file path.
        """
        self.logger.info("Starting inference...")
        gen_kwargs = self._build_gen_kwargs(overrides)
        result = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **gen_kwargs
        )
        self.logger.info("Inference completed.")
        waveform = torch.tensor(result["wav"])
        samplerate = default_xtts_settings["samplerate"]

        if save_path:
            try:
                torchaudio.save(save_path, waveform.unsqueeze(0), samplerate)
                self.logger.info(f"Audio saved to {save_path}")
                return {"file": save_path, "waveform": waveform, "samplerate": samplerate}
            except Exception as e:
                self.logger.error(f"Error saving audio: {e}")
                raise e
        else:
            return {"waveform": waveform, "samplerate": samplerate}

    def inference_stream(self, text: str, language: str, gpt_cond_latent, speaker_embedding, **overrides):
        """
        Generate speech audio in a streaming fashion.

        :param text: The text to synthesize.
        :param language: The language code for synthesis.
        :param gpt_cond_latent: Conditioning latent from reference audio.
        :param speaker_embedding: Speaker embedding for voice cloning.
        :param overrides: Generation parameter overrides.
        :return: A generator yielding chunks of audio tensors.
        """
        self.logger.info("Starting streaming inference...")
        gen_kwargs = self._build_gen_kwargs(overrides)
        return self.model.inference_stream(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **gen_kwargs
        )

    def save_audio(self, waveform: torch.Tensor, samplerate: int, file_path: str) -> str:
        """
        Save the provided audio waveform to a WAV file.

        :param waveform: The audio waveform tensor.
        :param samplerate: The sample rate for the audio.
        :param file_path: The destination file path.
        :return: The file path of the saved audio.
        """
        try:
            torchaudio.save(file_path, waveform.unsqueeze(0), samplerate)
            self.logger.info(f"Audio successfully saved to {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving audio file: {e}")
            raise e

    def get_conditioning_latents(self, audio_path: str):
        """
        Compute speaker conditioning latents from a reference audio file.

        :param audio_path: Path to the reference audio file.
        :return: A tuple (gpt_cond_latent, speaker_embedding).
        """
        self.logger.info("Computing speaker conditioning latents...")
        return self.model.get_conditioning_latents(audio_path=[audio_path])


    
    