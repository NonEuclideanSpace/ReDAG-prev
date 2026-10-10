#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python3 tests/test_imports.py
python3 tests/test_schema.py
python3 tests/test_labels.py
python3 tests/test_vlm_parse.py
python3 tests/test_planner_offline.py
python3 tests/test_dedupe.py
python3 tests/test_commit.py
python3 tests/test_credit_grpo.py
python3 tests/test_loop_mock.py
python3 tests/test_train_mock.py
python3 tests/test_scripts_mock.py
python3 tests/test_queue_mock.py
echo ALL_TESTS_OK
