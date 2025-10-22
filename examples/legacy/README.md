# Legacy Files

This directory contains outdated experimental files that are no longer actively used in the project.

## Files

### experiments.py (16KB)
**Date**: 2025-10-15
**Purpose**: Early experiment configuration system
**Status**: ⚠️ Outdated

Contains baseline model configurations for:
- GPT-4o
- Claude-3.5-Sonnet
- Llama-3.1-70B

**Why deprecated**:
- Current evaluation scripts (in `scripts/`) are more focused
- No longer using wandb experiment tracking
- API keys not configured

**Can be deleted**: Yes, unless planning to add API-based baselines

---

### grpo_table_qa.py (25KB)
**Date**: 2025-10-15
**Purpose**: Full GRPO training implementation
**Status**: ⚠️ Reference implementation

Contains complete implementation of:
- GRPO training loop
- Multi-component reward function
- Policy network integration

**Why moved to legacy**:
- GRPO is marked as TODO in project (user implementation with TRL)
- Can serve as reference but not actively maintained
- Actual GRPO interface is in `src/grpo/grpo_trainer.py`

**Can be deleted**: Keep as reference for future GRPO implementation

---

## Recommendation

- **experiments.py**: Can be safely deleted
- **grpo_table_qa.py**: Keep as reference for GRPO implementation

If you need to restore these files, they are tracked in git history.
