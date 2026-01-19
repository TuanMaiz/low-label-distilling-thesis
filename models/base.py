"""
Base model class for multilingual name entity matching.

Defines the model-agnostic interface for all seq2seq models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


class NameTranslationDataset(Dataset):
    """
    PyTorch Dataset for name translation task.

    Each item is a (input_text, target_text) pair.
    """

    def __init__(
        self,
        input_texts: List[str],
        target_texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        if len(input_texts) != len(target_texts):
            raise ValueError("input_texts and target_texts must have same length")

        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.input_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }


class BaseModel(ABC):
    """
    Abstract base class for all seq2seq models used in this project.

    All model implementations (mBART, NLLB, mT5, Aya) must inherit from this.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 128,
        device: Optional[str] = None
    ):
        """
        Initialize the base model.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        pass

    @abstractmethod
    def format_input(
        self,
        name: str,
        source_lang: str,
        target_lang: str,
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> str:
        """
        Format input for the model.

        Args:
            name: Source language name
            source_lang: Source language code
            target_lang: Target language code
            age: Optional age
            gender: Optional gender

        Returns:
            Formatted input string
        """
        pass

    @abstractmethod
    def format_target(self, name: str, lang: str) -> str:
        """
        Format target for the model.

        Args:
            name: Target language name
            lang: Target language code

        Returns:
            Formatted target string
        """
        pass

    def create_dataset(
        self,
        input_texts: List[str],
        target_texts: List[str]
    ) -> NameTranslationDataset:
        """Create a PyTorch dataset."""
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return NameTranslationDataset(
            input_texts=input_texts,
            target_texts=target_texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

    def create_dataloader(
        self,
        dataset: NameTranslationDataset,
        batch_size: int = 8,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch dataloader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True if self.device == "cuda" else False
        )

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_length: int = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a translation for a single input.

        Args:
            input_text: Input text
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if max_length is None:
            max_length = self.max_length

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        self.model.eval()
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    @torch.no_grad()
    def generate_batch(
        self,
        input_texts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate translations for a batch of inputs.

        Args:
            input_texts: List of input texts
            batch_size: Batch size for generation
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        generated_texts = []

        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            self.model.eval()
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                **kwargs
            )

            batch_generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts.extend(batch_generated)

        return generated_texts

    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_from_checkpoint(self, path: str) -> None:
        """Load model and tokenizer from a checkpoint."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.model.to(self.device)
