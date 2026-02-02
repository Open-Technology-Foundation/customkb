#!/usr/bin/env python
"""Backward-compatibility shim — all logging now lives in utils.logging_config."""

from utils.logging_config import (  # noqa: F401
  OperationLogger,
  dashes,
  elapsed_time,
  get_kb_info_from_config,
  get_log_file_path,
  get_logger,
  load_logging_config,
  log_file_operation,
  log_model_operation,
  log_operation_error,
  log_performance_metrics,
  setup_logging,
  time_to_finish,
)

#fin
