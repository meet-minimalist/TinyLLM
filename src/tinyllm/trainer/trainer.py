"""
Trainer — main training loop with proper state management.
"""

import torch
from torchinfo import summary

from src.tinyllm.callbacks.callback_handler import CallbackHandler
from src.tinyllm.logger.logger_utils import logger
from src.tinyllm.loss_fn.loss_helper import compute_ce_loss


class Trainer:
    """Handles the full training loop with callbacks, AMP, and gradient accumulation."""

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        tokenizer,
        train_loader,
        test_loader,
        train_config,
        model_config,
        device,
        callbacks,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_config = train_config
        self.model_config = model_config
        self.device = torch.device(device)
        self.callback_handler = CallbackHandler(callbacks)

        # AMP scaler
        self.use_amp = getattr(train_config, "fp16_training", False)
        if self.use_amp:
            from torch.amp import GradScaler

            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.grad_accum = getattr(train_config, "use_grad_accum", False)
        self.iters_to_accumulate = getattr(
            train_config, "iters_to_accumulate", 1
        )
        self.log_every = getattr(train_config, "log_every", 10)

        # Move model to device
        self.model.to(self.device)

        # Compile
        self.model = torch.compile(
            self.model,
            mode="max-autotune",  # best performance
            fullgraph=False,  # safer default
        )

        self._log_model_summary()

    def _log_model_summary(self):
        input_ids = torch.zeros(
            2,
            self.train_config.max_seq_len,
            dtype=torch.int32,
            device=self.device,
        )
        attn_mask = torch.zeros(
            2,
            self.train_config.max_seq_len,
            dtype=torch.int32,
            device=self.device,
        )
        summary(self.model, input_data=[input_ids, attn_mask], verbose=0)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Training {self.train_config.model_type} model with {n_params:,} params."
        )

    def train(self):
        self.callback_handler.on_train_begin(
            train_config=self.train_config,
            model_config=self.model_config,
            model=self.model,
        )

        self.global_step = 0

        for epoch in range(self.train_config.num_epochs):
            self._run_epoch(epoch)

        self.callback_handler.on_train_end()

    def _run_epoch(self, epoch: int):
        self.callback_handler.on_epoch_begin(
            epoch=epoch, global_step=self.global_step
        )

        # Training phase
        self._epoch_train()

        # Evaluation phase
        eval_loss, eval_ppl = self._epoch_eval()

        self.callback_handler.on_epoch_end(
            epoch=epoch,
            global_step=self.global_step,
            test_loss=eval_loss,
            test_ppl=eval_ppl,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scaler=self.scaler.state_dict() if self.scaler else None,
            metrics={
                "epoch": epoch,
                "eval_loss": eval_loss,
                "eval_ppl": eval_ppl,
            },
        )

        logger.info(
            f"Epoch {epoch + 1}/{self.train_config.num_epochs} — "
            f"Eval Loss: {eval_loss:.4f}, Eval PPL: {eval_ppl:.4f}"
        )

    def _epoch_train(self) -> tuple[float, float]:
        self.model.train()

        for batch_idx, (input_ids, attn_mask, labels) in enumerate(
            self.train_loader
        ):
            self._step_train(
                input_ids, attn_mask, labels, batch_idx, len(self.train_loader)
            )

    def _step_train(
        self, input_ids, attn_mask, labels, batch_idx, total_batches, should_log
    ) -> tuple[float, float]:
        input_ids = input_ids.to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        self.callback_handler.on_train_step_begin(
            global_step=self.global_step, batch_idx=batch_idx
        )

        # Forward pass
        if self.use_amp:
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16
            ):
                logits = self.model(input_ids, attn_mask)
                loss = compute_ce_loss(
                    logits,
                    labels,
                    self.train_config.get("label_smoothing", 0.0),
                    self.tokenizer.pad_token_id,
                )
                loss = loss / self.iters_to_accumulate
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids, attn_mask)
            loss = compute_ce_loss(
                logits,
                labels,
                self.train_config.get("label_smoothing", 0.0),
                self.tokenizer.pad_token_id,
            )
            loss.backward()

        # Gradient accumulation + optimizer step
        should_step = (not self.grad_accum) or (
            (batch_idx + 1) % self.iters_to_accumulate == 0
            or (batch_idx + 1 == total_batches)
        )

        if should_step:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # LR scheduler steps only on optimizer step
            self._step_lr()

        # Logging
        if (batch_idx + 1) % self.log_every == 0 or (
            batch_idx + 1 == total_batches
        ):
            lr = self.optimizer.param_groups[0]["lr"]
            ppl = torch.exp(loss * self.iters_to_accumulate).item()
            loss_value = loss.item() * self.iters_to_accumulate

            logger.info(
                f"Epoch step: {batch_idx + 1}/{total_batches}, "
                f"Loss: {loss_value:.4f}, PPL: {ppl:.4f}, LR: {lr:.6f}"
            )

            self.callback_handler.on_train_step_end(
                global_step=self.global_step,
                metrics={
                    "train_loss": loss_value,
                    "train_ppl": ppl,
                    "lr": lr,
                },
            )

        self.global_step += 1

    def _step_lr(self):
        """Step the LR scheduler."""
        if hasattr(self.lr_scheduler, "step"):
            self.lr_scheduler.step()

    def _epoch_eval(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_ppl = 0.0
        n_steps = 0

        with torch.no_grad():
            for input_ids, attn_mask, labels in self.test_loader:
                step_loss, step_ppl = self._step_eval(
                    input_ids, attn_mask, labels
                )
                total_loss += step_loss
                total_ppl += step_ppl
                n_steps += 1

        return total_loss / max(n_steps, 1), total_ppl / max(n_steps, 1)

    def _step_eval(self, input_ids, attn_mask, labels) -> tuple[float, float]:
        input_ids = input_ids.to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits = self.model(input_ids, attn_mask)
        loss = compute_ce_loss(logits, labels, 0.0, self.tokenizer.pad_token_id)
        ppl = torch.exp(loss).item()
        return loss.item(), ppl
