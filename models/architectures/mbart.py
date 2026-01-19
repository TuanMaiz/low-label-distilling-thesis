"""
mBART model implementation for multilingual name translation.

mBART (Multilingual Denoising Pre-training) is a sequence-to-sequence
denoising autoencoder pretrained on many languages.
It was the base model used in the original mGENRE paper.
"""

import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from typing import Optional

from models.base import BaseModel


# Language code mappings for mBART
MBART_LANG_CODES = {
    "ru": "ru_RU",
    "en": "en_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    "es": "es_XX",
    "it": "it_IT",
    "zh": "zh_CN",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "ar": "ar_AR",
    "hi": "hi_IN",
    "pt": "pt_XX",
    "tr": "tr_TR",
    "pl": "pl_PL",
    "nl": "nl_XX",
    "sv": "sv_SE",
    "fi": "fi_FI",
    "vi": "vi_VN",
    "th": "th_TH",
    "uk": "uk_UA",
    "be": "be_BY",
    "kk": "kk_KZ",
    "uz": "uz_UZ",
}


class MBartModel(BaseModel):
    """
    mBART model for multilingual name translation.

    Uses special tokens for task specification and language prefixes.
    """

    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        max_length: int = 128,
        device: Optional[str] = None
    ):
        """
        Initialize mBART model.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            device: Device to use
        """
        super().__init__(model_name, max_length, device)

    def load_model(self) -> None:
        """Load the mBART model and tokenizer."""
        self.tokenizer = Mbart50TokenizerFast.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)

    def _set_source_language(self, lang_code: str) -> None:
        """Set the source language for tokenization."""
        mbart_lang = MBART_LANG_CODES.get(lang_code, lang_code)
        self.tokenizer.src_lang = mbart_lang

    def _get_target_lang_token(self, lang_code: str) -> str:
        """Get the forced target language token for generation."""
        mbart_lang = MBART_LANG_CODES.get(lang_code, lang_code)
        # mBART uses language tokens like "ru_RU", "en_XX"
        # We need to convert to token id for forced decoding
        return self.tokenizer.lang_code_to_id[mbart_lang]

    def format_input(
        self,
        name: str,
        source_lang: str,
        target_lang: str,
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> str:
        """
        Format input for mBART.

        Format: [TRANSLATE] [SOURCE→TARGET] [AGE:XX] [GENDER:X] <name>

        Args:
            name: Source language name
            source_lang: Source language code (e.g., "ru")
            target_lang: Target language code (e.g., "en")
            age: Optional age
            gender: Optional gender ("M" or "F")

        Returns:
            Formatted input string
        """
        parts = []
        parts.append("[TRANSLATE]")
        parts.append(f"[{source_lang.upper()}→{target_lang.upper()}]")

        if age is not None:
            parts.append(f"[AGE:{age}]")
        if gender is not None:
            parts.append(f"[GENDER:{gender}]")

        parts.append(name)

        return " ".join(parts)

    def format_target(self, name: str, lang: str) -> str:
        """
        Format target for mBART.

        For mBART, we include the language prefix in the target
        to help the model learn the target language.

        Args:
            name: Target language name
            lang: Target language code

        Returns:
            Formatted target string with language prefix
        """
        return f"[{lang.upper()}] {name}"

    def generate(
        self,
        input_text: str,
        source_lang: str = "ru",
        target_lang: str = "en",
        max_length: int = None,
        num_beams: int = 5,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a translation with mBART.

        Overrides the base method to handle mBART's language-specific setup.

        Args:
            input_text: Input text
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            **kwargs: Additional generation arguments

        Returns:
            Generated text (without language prefix)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if max_length is None:
            max_length = self.max_length

        # Set source language for tokenization
        self._set_source_language(source_lang)

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Get target language token for forced decoding
        target_lang_id = self._get_target_lang_token(target_lang)

        # Generate
        self.model.eval()
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            forced_bos_token_id=target_lang_id,
            **kwargs
        )

        # Decode and clean up (remove language prefix if present)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove language prefix if model generated it
        for lang in ["en", "ru", "de", "fr", "es", "it"]:
            prefix = f"[{lang.upper()}] "
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):]
                break

        return generated_text.strip()

    @torch.no_grad()
    def generate_with_score(
        self,
        input_text: str,
        source_lang: str = "ru",
        target_lang: str = "en",
        max_length: int = None,
        num_beams: int = 5,
        **kwargs
    ) -> tuple[str, float]:
        """
        Generate a translation and return the model's confidence score.

        The score is the average token probability (log-prob converted to probability).

        Args:
            input_text: Input text
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            Tuple of (generated_text, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if max_length is None:
            max_length = self.max_length

        self._set_source_language(source_lang)

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        target_lang_id = self._get_target_lang_token(target_lang)

        self.model.eval()

        # Generate with output scores
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            forced_bos_token_id=target_lang_id,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Remove language prefix if present
        for lang in ["en", "ru", "de", "fr", "es", "it"]:
            prefix = f"[{lang.upper()}] "
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):]
                break

        # Calculate average probability from scores
        # outputs.scores is a list of tensors, one for each generated token
        if outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                # score is (batch, vocab_size) - get log prob for the chosen token
                token_id = outputs.sequences[0, i + 1]  # +1 because first token is BOS
                log_prob = score[0, token_id].item()
                log_probs.append(log_prob)

            # Convert log probs to probabilities and average
            avg_log_prob = sum(log_probs) / len(log_probs)
            confidence_score = avg_log_prob  # Keep as log prob for now
        else:
            confidence_score = float("-inf")

        return generated_text.strip(), confidence_score


if __name__ == "__main__":
    # Test the mBART model
    print("Testing mBART model...")

    model = MBartModel()

    print(f"Model: {model.model_name}")
    print(f"Device: {model.device}")

    # Test formatting
    input_text = model.format_input(
        name="Владимир Путин",
        source_lang="ru",
        target_lang="en",
        age=68,
        gender="M"
    )
    print(f"\nInput: {input_text}")

    target_text = model.format_target(name="Vladimir Putin", lang="en")
    print(f"Target: {target_text}")

    # Note: Actually loading the model requires downloading ~2.5GB
    # Uncomment to test with actual model:
    # model.load_model()
    # print(f"\nParameters: {model.get_num_parameters():,}")
