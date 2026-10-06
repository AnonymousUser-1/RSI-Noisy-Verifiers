"""Keep an exported RSI_BASE_PIN out of the tests that check the study's own pin.

rsi.base_pin.default_pin_path() prefers RSI_BASE_PIN to BASE_PIN, so a shell that exports it for
another base (README: a Llama or Qwen3-4B study) would redirect every test that relies on the
frozen pin or patches BASE_PIN.  A module that needs the study's pin imports these two as its
setUpModule and tearDownModule; pytest and unittest both run those (pytest does not run
unittest.addModuleCleanup).  Tests that need another pin set RSI_BASE_PIN themselves.
"""
import os
from unittest import mock

_environ = None


def isolate():
    global _environ
    _environ = mock.patch.dict(os.environ)
    _environ.start()
    os.environ.pop("RSI_BASE_PIN", None)


def restore():
    _environ.stop()
