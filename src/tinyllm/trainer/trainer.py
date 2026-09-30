import contextlib
import math
import signal
import time

import torch

from src.tinyllm.callbacks.callback_handler import CallbackHandler
from src.tinyllm.logger.logger_utils import logger
from src.tinyllm.loss_fn.loss_helper import compute_ce_loss
from src.tinyllm.utils.checkpoint import capture_rng, restore_rng
from src.tinyllm.utils.distributed import DistInfo, all_reduce
from src.tinyllm.utils.model_stats import peak_flops
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
    """Training loop. One ``step`` is one optimizer update.

    A step consumes ``iters_to_accumulate`` micro-batches on every rank, so the
    tokens per step are ``packed_tokens * iters_to_accumulate * world_size``.
    ``num_training_steps``, the LR schedule, logging and checkpoints all count
    optimizer steps.

    Resume: ``state_dict()`` / ``load_state_dict()`` cover model, optimizer,
    LR scheduler, GradScaler, counters, data position and RNG. The data
    position is the number of micro-batches this rank consumed in the current
    epoch; ``make_train_loader(skip_batches)`` rebuilds the loader past them.
    """

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        tokenizer,
        make_train_loader,
        test_loader,
        train_config,
        model_config,
        device,
        callbacks,
        dist_info: DistInfo | None = None,
        ckpt_manager=None,
        resume_state: dict | None = None,
        flops_per_token: float | None = None,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.make_train_loader = make_train_loader
        self.test_loader = test_loader
        self.train_config = train_config
        self.model_config = model_config
        self.device = torch.device(device)
        self.callback_handler = CallbackHandler(callbacks)
        self.dist = dist_info or DistInfo()
        self.ckpt_manager = ckpt_manager
        self.flops_per_token = flops_per_token

        # precision: "bf16" (default) | "fp16" | "fp32"
        self.precision = train_config.get("precision", "bf16")
        # GradScaler only needed for fp16 — bf16 has fp32 range, never overflows.
        if self.precision == "fp16":
            from torch.amp import GradScaler

            self.scaler = GradScaler(self.device.type)
        else:
            self.scaler = None

        # Fused linear+CE loss — avoids materializing [S, vocab] logits.
        # Tries Liger (Linux) then cut-cross-entropy (Windows); falls back to
        # standard CE. Both fused kernels are GPU-only.
        from src.tinyllm.utils.kernels import try_fused_ce

        label_smoothing = train_config.get("label_smoothing", 0.0)
        kernels_cfg = train_config.get("kernels", {}) or {}
        self._fused_ce, _backend = (None, None)
        if self.device.type == "cuda":
            self._fused_ce, _backend = try_fused_ce(
                ignore_index=IGNORE_INDEX,
                label_smoothing=label_smoothing,
                use_liger=kernels_cfg.get("use_liger", False),
            )
        if self._fused_ce is not None:
            logger.info(f"Using fused linear+CE loss (backend: {_backend})")
        else:
            logger.info("Fused CE not available; using standard cross-entropy.")

        self.iters_to_accumulate = (
            train_config.get("iters_to_accumulate", 1)
            if train_config.get("use_grad_accum", False)
            else 1
        )
        self.max_grad_norm = train_config.get("max_grad_norm", None)
        self.log_every = train_config.get("log_every", 10)
        self.max_steps = train_config.get("num_training_steps", 0) or 0
        self.num_epochs = train_config.get("num_epochs", 1)
        self.eval_every_steps = train_config.get("eval_every_steps", 0) or 0
        self.eval_max_batches = train_config.get("eval_max_batches", 0) or 0
        self.save_every_steps = train_config.get("save_every_steps", 0) or 0
        self.save_every_minutes = train_config.get("save_every_minutes", 0) or 0
        # Stop (and save) before a hard session limit, e.g. Kaggle's 12 h.
        self.max_runtime_minutes = (
            train_config.get("max_runtime_minutes", 0) or 0
        )

        # Counters. All of these are saved in checkpoints.
        self.step = 0  # optimizer steps
        self.tokens = 0  # tokens trained on, all ranks
        self.epoch = 0
        self.batches_in_epoch = 0  # micro-batches this rank consumed this epoch
        self._consecutive_nonfinite = 0
        self._last_saved_step = None
        self._last_eval = None  # (step, loss, ppl) of the latest eval

        self._stop_signal = False
        self._install_signal_handlers()

        # Load weights before DDP wraps the model, so every rank starts equal.
        self.raw_model = model.to(self.device)
        if resume_state is not None:
            self.load_state_dict(resume_state)

        self.model = self.raw_model
        if self.dist.enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP

            # broadcast_buffers=False: the only buffers are deterministic RoPE /
            # sinusoidal tables. Broadcasting them would add a collective to
            # every forward, which hangs when ranks run different numbers of
            # eval batches.
            self.model = DDP(
                self.raw_model,
                device_ids=(
                    [self.dist.local_rank]
                    if self.device.type == "cuda"
                    else None
                ),
                broadcast_buffers=False,
            )

        use_compile = train_config.get("use_compile", True)
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

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict()
                if hasattr(self.lr_scheduler, "state_dict")
                else None
            ),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "trainer": {
                "step": self.step,
                "tokens": self.tokens,
                "epoch": self.epoch,
                "batches_in_epoch": self.batches_in_epoch,
                "world_size": self.dist.world_size,
                "iters_to_accumulate": self.iters_to_accumulate,
            },
            "rng": capture_rng(),
            "wandb_run_id": self.train_config.get("resume_wandb_id"),
            "train_config": self.train_config.to_dict(),
            "model_config": self.model_config.to_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("lr_scheduler") and hasattr(
            self.lr_scheduler, "load_state_dict"
        ):
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if self.scaler is not None and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        t = state["trainer"]
        self.step = t["step"]
        self.tokens = t["tokens"]
        self.epoch = t["epoch"]
        self.batches_in_epoch = t["batches_in_epoch"]
        if t.get("world_size", 1) != self.dist.world_size:
            # The data position is per rank, so it only maps 1:1 onto the same
            # world size. With a different one, start the epoch's data over
            # from this rank's share instead of silently mis-skipping.
            logger.warning(
                f"Checkpoint was written with world_size={t.get('world_size')} "
                f"but this run has {self.dist.world_size}; the data position "
                f"cannot be mapped exactly. Skipping "
                f"{self.batches_in_epoch * t.get('world_size', 1) // self.dist.world_size} "
                f"batches per rank (approximate)."
            )
            self.batches_in_epoch = (
                self.batches_in_epoch
                * t.get("world_size", 1)
                // self.dist.world_size
            )
        restore_rng(state.get("rng"))
        logger.info(
            f"Resumed at step {self.step:,} (epoch {self.epoch}, "
            f"{self.tokens:,} tokens, {self.batches_in_epoch:,} batches into the epoch)"
        )

    def save_checkpoint(self) -> None:
        if self.ckpt_manager is not None and self._last_saved_step != self.step:
            self.ckpt_manager.save(self.state_dict(), self.step)
            self._last_saved_step = self.step

    # ---------------------------------------------------------------- signals
    def _install_signal_handlers(self):
        """SIGTERM/SIGINT ask for a clean stop: save, then exit."""

        def _handler(signum, frame):
            logger.warning(
                f"Signal {signum} received: will save and stop after this step."
            )
            self._stop_signal = True

        try:
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, AttributeError):  # not main thread / platform
            pass

    def _time_up(self) -> bool:
        if not self.max_runtime_minutes:
            return False
        return (
            time.time() - self._start_time
        ) / 60.0 >= self.max_runtime_minutes

    def _sync_flags(self, *flags: bool) -> list[bool]:
        """OR each flag across ranks, so every rank takes the same branch."""
        if not self.dist.enabled:
            return list(flags)
        t = torch.tensor([float(f) for f in flags], device=self.device)
        all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return [bool(v > 0) for v in t.tolist()]

    # ------------------------------------------------------------------- loop
    def train(self):
        self._start_time = time.time()
        self._last_save_time = self._start_time
        self._log_t0 = time.time()
        self._log_tokens0 = self.tokens

        self.callback_handler.on_train_begin(
            train_config=self.train_config,
            model_config=self.model_config,
            model=self.raw_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        stopped_early = False
        while self.epoch < self.num_epochs:
            outcome = self._train_epoch()
            if outcome == "stop":
                stopped_early = True
                break
            if outcome == "max_steps":
                break
            # Data exhausted: end of epoch.
            self._end_epoch()
            self.epoch += 1
            self.batches_in_epoch = 0

        if stopped_early:
            self.save_checkpoint()
            logger.info(
                f"Stopped at step {self.step:,} to respect the time limit / "
                f"signal. Run the same command again to resume."
            )
        else:
            if self.epoch < self.num_epochs:  # ended on max_steps
                self._end_epoch()
            self.save_checkpoint()
            logger.info(f"Training complete at step {self.step:,}.")

        if self.ckpt_manager is not None:
            self.ckpt_manager.wait()
        self.callback_handler.on_train_end()

    def _train_epoch(self) -> str:
        """Returns "exhausted", "max_steps" or "stop"."""
        self.model.train()
        data_iter = iter(self.make_train_loader(self.batches_in_epoch))
        while True:
            if self.max_steps and self.step >= self.max_steps:
                logger.info(f"Reached {self.max_steps:,} steps.")
                return "max_steps"

            batches = []
            for _ in range(self.iters_to_accumulate):
                try:
                    batches.append(next(data_iter))
                except StopIteration:
                    break

            save_due = bool(
                self.save_every_minutes
                and (time.time() - self._last_save_time) / 60.0
                >= self.save_every_minutes
            )
            exhausted, stop, save_due = self._sync_flags(
                len(batches) < self.iters_to_accumulate,
                self._stop_signal or self._time_up(),
                save_due,
            )
            if exhausted:
                # A partial accumulation window at the end of the data is
                # dropped, so every step has the same token count.
                return "exhausted"
            if stop:
                return "stop"

            self._train_step(batches)

            if save_due or (
                self.save_every_steps and self.step % self.save_every_steps == 0
            ):
                self.save_checkpoint()
                self._last_save_time = time.time()
            if self.eval_every_steps and self.step % self.eval_every_steps == 0:
                self._run_eval(tag="eval")
                self.model.train()

    def _train_step(self, batches):
        self.callback_handler.on_train_step_begin(
            global_step=self.step, batch_idx=self.batches_in_epoch
        )

        loss_sum = torch.zeros((), device=self.device)
        local_tokens = 0
        for i, batch in enumerate(batches):
            last = i == len(batches) - 1
            # Skip the DDP gradient all-reduce on all but the last micro-batch.
            sync_ctx = (
                self.model.no_sync()
                if self.dist.enabled
                and not last
                and hasattr(self.model, "no_sync")
                else contextlib.nullcontext()
            )
            with sync_ctx:
                loss, n_tokens = self._forward_loss(batch)
                loss = loss / len(batches)
                # Always run backward, even for a non-finite loss: under DDP a
                # rank that skipped backward would leave the others waiting in
                # the all-reduce forever. The NaN reaches the gradient norm,
                # which is identical on every rank, and all ranks skip together.
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            loss_sum += loss.detach()
            local_tokens += n_tokens
        self.batches_in_epoch += len(batches)

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.raw_model.parameters(),
            self.max_grad_norm if self.max_grad_norm else float("inf"),
        )

        finite = bool(torch.isfinite(grad_norm))
        if self.scaler is not None:
            # fp16: an overflow is routine. scaler.step skips the update and
            # scaler.update lowers the scale; skipping both would keep the scale
            # too high and overflow on every step after. Only a long run of
            # overflows means the model itself is broken.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if not finite:
                self._consecutive_nonfinite += 1
                if self._consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
                    self._report_nonfinite(loss_sum, grad_norm, batches)
                self.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.lr_scheduler, "step"):
                    self.lr_scheduler.step()
                self.step += 1
                self.tokens += local_tokens * self.dist.world_size
                return
        elif not finite:
            self.optimizer.zero_grad(set_to_none=True)
            self._report_nonfinite(loss_sum, grad_norm, batches)
            return
        else:
            self.optimizer.step()
        self._consecutive_nonfinite = 0
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(self.lr_scheduler, "step"):
            self.lr_scheduler.step()

        self.step += 1
        # Every rank processes the same token count per micro-batch in packed
        # mode, so the global count is local * world_size.
        self.tokens += local_tokens * self.dist.world_size

        if self.step % self.log_every == 0:
            self._log_step(loss_sum, grad_norm)

    def _forward_loss(self, batch):
        """Returns (mean loss over the batch's tokens, number of tokens)."""
        if len(batch) == 3:
            inputs, targets, cu_seqlens = batch
        else:
            (inputs, targets), cu_seqlens = batch, None

        # Move to device here (not in dataset) so pin_memory + non_blocking
        # can overlap transfer with GPU compute when num_workers > 0.
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        with get_autocast_ctx(self.device, self.precision):
            if self._fused_ce is not None:
                # Fused path: hidden states → lm_head + softmax + CE in one kernel.
                # Avoids materializing [S, vocab_size] logits.
                hidden = self.model(
                    inputs, cu_seqlens=cu_seqlens, return_hidden=True
                )
                B, S, D = hidden.shape
                # RMSNorm outputs fp32 even under autocast; CCE backward requires bf16/fp16.
                amp_dtype = (
                    torch.bfloat16
                    if self.precision == "bf16"
                    else torch.float16
                )
                loss = self._fused_ce(
                    hidden.view(B * S, D).to(amp_dtype),
                    self.raw_model.lm_head.weight,
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
        return loss, inputs.numel()

    # ---------------------------------------------------------------- logging
    def _log_step(self, loss_sum, grad_norm):
        loss_t = loss_sum.detach().clone()
        all_reduce(loss_t)
        loss_value = loss_t.item() / self.dist.world_size
        ppl = math.exp(min(loss_value, 50.0))
        lr = self.optimizer.param_groups[0]["lr"]

        now = time.time()
        dt = max(now - self._log_t0, 1e-9)
        tok_per_s = (self.tokens - self._log_tokens0) / dt
        self._log_t0, self._log_tokens0 = now, self.tokens

        metrics = {
            "train_loss": loss_value,
            "train_ppl": ppl,
            "lr": lr,
            "tokens": self.tokens,
            "grad_norm": float(grad_norm),
            "tokens_per_sec": tok_per_s,
        }
        mfu_str = ""
        if self.flops_per_token and self.device.type == "cuda":
            peak = peak_flops(torch.cuda.get_device_name(self.device))
            if peak:
                mfu = (
                    self.flops_per_token
                    * tok_per_s
                    / (peak * self.dist.world_size)
                )
                metrics["mfu"] = mfu
                mfu_str = f", MFU: {mfu:.1%}"
        if self.scaler is not None:
            metrics["loss_scale"] = self.scaler.get_scale()

        total = f"/{self.max_steps:,}" if self.max_steps else ""
        logger.info(
            f"Step: {self.step:,}{total}, Loss: {loss_value:.4f}, "
            f"PPL: {ppl:.2f}, LR: {lr:.6f}, GradNorm: {float(grad_norm):.3f}, "
            f"Tok/s: {tok_per_s:,.0f}{mfu_str}, Tokens: {self.tokens:,}"
        )

        self.callback_handler.on_train_step_end(
            global_step=self.step,
            metrics=metrics,
            model=self.raw_model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def _report_nonfinite(self, loss_sum, grad_norm, batches):
        """Log enough to identify the offending batch, then decide whether to
        keep going. Skipping is only safe for isolated batches — a run that
        cannot produce a finite loss is burning GPU time, so abort once the
        failures become consecutive."""
        self._consecutive_nonfinite += 1

        details = [
            f"step={self.step}",
            f"batches_in_epoch={self.batches_in_epoch}",
            f"loss={loss_sum.item()}",
            f"grad_norm={grad_norm.item()}",
        ]
        for i, batch in enumerate(batches):
            inputs = batch[0]
            details.append(
                f"mb{i}: inputs={tuple(inputs.shape)} "
                f"token_id_range=[{inputs.min().item()}, {inputs.max().item()}]"
            )
            if len(batch) == 3:
                cu = batch[2]
                seg_lens = (cu[1:] - cu[:-1]).tolist()
                details.append(
                    f"mb{i}: segments={len(seg_lens)} "
                    f"len=[{min(seg_lens)}, {max(seg_lens)}] "
                    f"zero_length={sum(1 for L in seg_lens if L == 0)}"
                )

        logger.error(
            f"Non-finite loss/grad — skipping step "
            f"({self._consecutive_nonfinite}/{MAX_CONSECUTIVE_NONFINITE} "
            f"consecutive). " + ", ".join(details)
        )

        if self._consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
            raise RuntimeError(
                f"{self._consecutive_nonfinite} consecutive non-finite steps "
                f"at step {self.step}. The model is very likely already "
                f"corrupted; stopping instead of training on garbage."
            )

    # ------------------------------------------------------------------- eval
    def _end_epoch(self):
        eval_loss, eval_ppl = self._run_eval(tag="epoch")
        self.callback_handler.on_epoch_end(
            epoch=self.epoch,
            global_step=self.step,
            test_loss=eval_loss,
            test_ppl=eval_ppl,
            metrics={
                "epoch": self.epoch,
                "eval_loss": eval_loss,
                "eval_ppl": eval_ppl,
            },
        )

    @torch.no_grad()
    def _run_eval(self, tag: str):
        """Mean loss over the (rank-sharded) test loader, averaged over ranks."""
        if self.test_loader is None:
            return float("nan"), float("nan")
        if self._last_eval is not None and self._last_eval[0] == self.step:
            return self._last_eval[1], self._last_eval[2]
        self.model.eval()
        totals = torch.zeros(
            2, device=self.device
        )  # [sum of losses, n batches]
        for i, batch in enumerate(self.test_loader):
            if self.eval_max_batches and i >= self.eval_max_batches:
                break
            loss, _ = self._forward_loss(batch)
            totals[0] += loss.float()
            totals[1] += 1
        all_reduce(totals)
        avg_loss = (totals[0] / totals[1].clamp(min=1)).item()
        ppl = math.exp(min(avg_loss, 50.0))
        self._last_eval = (self.step, avg_loss, ppl)
        logger.info(
            f"[{tag}] step {self.step:,} — Eval Loss: {avg_loss:.4f}, "
            f"Eval PPL: {ppl:.2f}"
        )
        if tag == "eval":
            self.callback_handler.on_evaluate(
                global_step=self.step,
                metrics={"eval_loss": avg_loss, "eval_ppl": ppl},
            )
        return avg_loss, ppl
