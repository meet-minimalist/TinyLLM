from abc import ABC, abstractmethod


class BaseBenchmark(ABC):
    @abstractmethod
    def run(self, model, tokenizer, device):
        """Run benchmark evaluation.

        Args:
            model: nn.Module in eval mode.
            tokenizer: Tokenizer instance.
            device: torch device.

        Returns:
            Dict with at least {"accuracy": float}.
        """
        ...
