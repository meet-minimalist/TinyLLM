import torch
from torchinfo import summary

from src.tinyllm.callbacks.callback_handler import CallbackHandler
from src.tinyllm.logger.logger_utils import logger
from src.tinyllm.loss_fn.loss_helper import compute_ce_loss
from src.tinyllm.utils.train_utils import get_autocast_ctx

# The packer never pads — every position in a batch is a real token — so no
# label should be masked out. pad_token_id must NOT be used here: the GPT-2
# tokenizer has no pad token, so get_tokenizer aliases it to eos (50256), which
# is the document separator in the FineWeb data. Using it would silently drop
# every end-of-document target from the loss.
IGNORE_INDEX = -100

# Consecutive non-finite steps tolerated before the run is aborted.
MAX_CONSECUTIVE_NONFINITE = 20


class Trainer:
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

        # precision: "bf16" (default) | "fp16" | "fp32"
        self.precision = getattr(train_config, "precision", "bf16")
        # GradScaler only needed for fp16 — bf16 has fp32 range, never overflows.
        if self.precision == "fp16":
            from torch.amp import GradScaler

            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Fused linear+CE loss — avoids materializing [S, vocab] logits.
        # Tries Liger (Linux) then cut-cross-entropy (Windows); falls back to standard CE.
        from src.tinyllm.utils.kernels import try_fused_ce

        label_smoothing = (
            train_config.get("label_smoothing", 0.0)
            if hasattr(train_config, "get")
            else getattr(train_config, "label_smoothing", 0.0)
        )
        kernels_cfg = (
            train_config.get("kernels", {})
            if hasattr(train_config, "get")
            else getattr(train_config, "kernels", {})
        ) or {}
        self._fused_ce, _backend = try_fused_ce(
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
            use_liger=kernels_cfg.get("use_liger", False),
        )
        if self._fused_ce is not None:
            logger.info(f"Using fused linear+CE loss (backend: {_backend})")
        else:
            logger.info("Fused CE not available; using standard cross-entropy.")

        self.grad_accum = getattr(train_config, "use_grad_accum", False)
        self.iters_to_accumulate = getattr(
            train_config, "iters_to_accumulate", 1
        )
        self.max_grad_norm = getattr(train_config, "max_grad_norm", None)
        self.log_every = getattr(train_config, "log_every", 10)

        self.global_step = 0
        self.global_tokens = 0
        self._consecutive_nonfinite = 0
        # nGPT keeps its weights on the unit hypersphere by retracting after
        # every optimizer step rather than constraining the forward pass.
        self._normalizes_weights = getattr(model, "ngpt", False)

        self.model.to(self.device)

        use_compile = getattr(train_config, "use_compile", True)
        if use_compile and self.device.type == "cuda":
            # suppress_errors=True: allows graph breaks on ops torch.compile can't
            # trace (e.g. Liger's custom autograd functions in PyTorch 2.11).
            # Those ops run in eager; everything else is compiled.
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(
                self.model,
                mode="default",
                dynamic=True,  # cu_seqlens shape varies per batch (varlen packing)
            )
            logger.info(
                "Model compiled with torch.compile (default, dynamic=True)"
            )

        self._log_model_summary()

    def _log_model_summary(self):
        input_ids = torch.zeros(
            1,
            self.model_config.max_seq_len,
            dtype=torch.int32,
            device=self.device,
        )
        try:
            summary(self.model, input_data=[input_ids], verbose=0)
        except Exception:
            pass
        params = list(self.model.parameters())
        n_params = sum(p.numel() for p in params)
        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        # element_size() = bytes per element for the param's dtype (fp32 -> 4).
        param_bytes = sum(p.numel() * p.element_size() for p in params)

        def _fmt(n: int) -> str:
            for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
                if n >= scale:
                    return f"{n / scale:.2f}{unit}"
            return str(n)

        model_name = self.model_config.get("name") or self.model_config.get(
            "model_type", "experiment"
        )
        logger.info(
            f"Model: {model_name} — "
            f"{_fmt(n_params)} params ({n_params:,}), "
            f"{_fmt(n_trainable)} trainable, "
            f"{param_bytes / 1024 ** 2:.1f} MB"
        )

    def train(self):
        self.callback_handler.on_train_begin(
            train_config=self.train_config,
            model_config=self.model_config,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        for epoch in range(self.train_config.num_epochs):
            self._run_epoch(epoch)

        self.callback_handler.on_train_end()

    def _run_epoch(self, epoch: int):
        self.callback_handler.on_epoch_begin(
            epoch=epoch, global_step=self.global_step
        )
        self._epoch_train()
        eval_loss, eval_ppl = self._epoch_eval()

        self.callback_handler.on_epoch_end(
            epoch=epoch,
            global_step=self.global_step,
            test_loss=eval_loss,
            test_ppl=eval_ppl,
            model=self.model.state_dict(),
            optimizer=(
                self.optimizer.state_dict()
                if hasattr(self.optimizer, "state_dict")
                else None
            ),
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

    def _epoch_train(self):
        self.model.train()
        max_steps = self.train_config.get("num_training_steps", 0) or 0
        if max_steps == 0 and self.train_config.get("num_epochs", 1) > 0:
            # run until dataloader exhausts or max_batches limit is hit
            pass
        try:
            total = len(self.train_loader)
        except TypeError:
            total = (
                self.train_config.get("max_batches", 0)
                or max_steps
                or 1_000_000_000
            )
        for batch_idx, batched_input in enumerate(self.train_loader):
            if max_steps and self.global_step >= max_steps:
                logger.info(f"Reached {max_steps} steps, stopping training")
                return
            self._step_train(batched_input, batch_idx, total)

    def _step_train(self, batched_input, batch_idx, total_batches):
        if len(batched_input) == 3:
            inputs, targets, cu_seqlens = batched_input
        else:
            inputs, targets = batched_input
            cu_seqlens = None

        # Move to device here (not in dataset) so pin_memory + non_blocking
        # can overlap transfer with GPU compute when num_workers > 0.
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        self.callback_handler.on_train_step_begin(
            global_step=self.global_step,
            batch_idx=batch_idx,
        )

        autocast_ctx = get_autocast_ctx(self.device, self.precision)
        with autocast_ctx:
            if self._fused_ce is not None:
                # Fused path: hidden states → lm_head + softmax + CE in one kernel.
                # Avoids materializing [S, vocab_size] logits (~100 MB for vocab=50304).
                hidden = self.model(
                    inputs, cu_seqlens=cu_seqlens, return_hidden=True
                )
                B, S, D = hidden.shape
                # nGPT folds its logit scale s_z into this matrix; the
                # baseline returns lm_head.weight unchanged.
                lm_weight = self.model.lm_head_weight()
                # RMSNorm outputs fp32 even under autocast; CCE backward requires bf16/fp16.
                amp_dtype = (
                    torch.bfloat16
                    if self.precision == "bf16"
                    else torch.float16
                )
                loss = self._fused_ce(
                    hidden.view(B * S, D).to(amp_dtype),
                    lm_weight,
                    targets.view(B * S),
                )
            else:
                logits = self.model(inputs, cu_seqlens=cu_seqlens)
                loss = compute_ce_loss(
                    logits,
                    targets,
                    self.train_config.get("label_smoothing", 0.0),
                    IGNORE_INDEX,
                )
            loss = loss / self.iters_to_accumulate

        # A non-finite loss poisons every parameter the moment optimizer.step()
        # runs, and it is unrecoverable: clip_grad_norm_ does not filter NaN
        # (error_if_nonfinite defaults to False), so the corruption spreads
        # silently and every later step trains on garbage. Skip the batch.
        if not torch.isfinite(loss):
            self._report_nonfinite("loss", loss, inputs, cu_seqlens, batch_idx)
            self.optimizer.zero_grad(set_to_none=True)
            return

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (not self.grad_accum) or (
            (batch_idx + 1) % self.iters_to_accumulate == 0
            or (batch_idx + 1 == total_batches)
        )

        if should_step:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                # The loss can be finite while the backward pass still produces
                # NaN/Inf grads, so check the norm clip_grad_norm_ already
                # computed rather than paying for a second pass.
                if not torch.isfinite(grad_norm):
                    self._report_nonfinite(
                        "grad_norm", grad_norm, inputs, cu_seqlens, batch_idx
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    return
            self._consecutive_nonfinite = 0
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self._normalizes_weights:
                self.model.normalize_weights()
            self.optimizer.zero_grad(set_to_none=True)
            self._step_lr()

        self.global_step += 1
        self.global_tokens += inputs.numel()

        if (batch_idx + 1) % self.log_every == 0 or (
            batch_idx + 1 == total_batches
        ):
            lr = (
                self.optimizer.param_groups[0]["lr"]
                if hasattr(self.optimizer, "param_groups")
                else 0
            )
            ppl = torch.exp(loss * self.iters_to_accumulate).item()
            loss_value = loss.item() * self.iters_to_accumulate
            logger.info(
                f"Step: {self.global_step}/{total_batches}, "
                f"Loss: {loss_value:.4f}, PPL: {ppl:.4f}, "
                f"LR: {lr:.6f}, Tokens: {self.global_tokens}"
            )

            self.callback_handler.on_train_step_end(
                global_step=self.global_step,
                metrics={
                    "train_loss": loss_value,
                    "train_ppl": ppl,
                    "lr": lr,
                    "tokens": self.global_tokens,
                },
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                tokenizer=self.tokenizer,
                device=self.device,
            )

    def _report_nonfinite(self, what, value, inputs, cu_seqlens, batch_idx):
        """Log everything needed to identify the offending batch, then decide
        whether to keep going. Skipping is only safe for isolated batches — a
        run that cannot produce a finite loss is burning GPU time, so abort
        once the failures become consecutive."""
        self._consecutive_nonfinite += 1

        details = [
            f"step={self.global_step}",
            f"batch_idx={batch_idx}",
            f"{what}={value.item()}",
            f"inputs={tuple(inputs.shape)}",
            f"token_id_range=[{inputs.min().item()}, {inputs.max().item()}]",
        ]
        if cu_seqlens is not None:
            seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            zero_at = [i for i, L in enumerate(seg_lens) if L == 0]
            details += [
                f"num_segments={len(seg_lens)}",
                f"segment_len_min={min(seg_lens)}",
                f"segment_len_max={max(seg_lens)}",
                f"zero_length_segments={len(zero_at)}",
            ]
            if zero_at:
                details.append(f"zero_length_at={zero_at[:16]}")

        logger.error(
            f"Non-finite {what} — skipping batch "
            f"({self._consecutive_nonfinite}/{MAX_CONSECUTIVE_NONFINITE} "
            f"consecutive). " + ", ".join(details)
        )

        if self._consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
            raise RuntimeError(
                f"{self._consecutive_nonfinite} consecutive non-finite steps "
                f"at step {self.global_step}. The model is very likely already "
                f"corrupted; stopping instead of training on garbage."
            )

    def _step_lr(self):
        if hasattr(self.lr_scheduler, "step"):
            self.lr_scheduler.step()

    def _epoch_eval(self):
        self.model.eval()
        total_loss = 0.0
        n_steps = 0
        autocast_ctx = get_autocast_ctx(self.device, self.precision)
        with torch.no_grad(), autocast_ctx:
            for batched_input in self.test_loader:
                # Unpack: varlen -> (inputs, targets, cu_seqlens), fixed -> (inputs, targets)
                if len(batched_input) == 3:
                    inputs, targets, cu_seqlens = batched_input
                else:
                    inputs, targets = batched_input
                    cu_seqlens = None
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if cu_seqlens is not None:
                    # flash_attn_varlen_func requires this on the same device;
                    # the training path already moves it.
                    cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
                logits = self.model(inputs, cu_seqlens=cu_seqlens)
                loss = compute_ce_loss(logits, targets, 0.0, IGNORE_INDEX)
                total_loss += loss.item()
                n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        return avg_loss, torch.exp(torch.tensor(avg_loss)).item()
