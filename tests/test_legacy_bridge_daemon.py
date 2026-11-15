"""Regression at the queued-task boundary for retired engine stamps."""

from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.core.core import CoreMessage, LoopStepStatus
from alphonse.agent_v2.core.io.channels import channel_metadata
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon, _scheduled_failure_message
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore




def test_scheduled_unsupported_work_explains_review_instead_of_retrying():
    message = _scheduled_failure_message("Reminder", "intelligence_engine_unregistered")
    assert "previous state needs review" in message
    assert "model" not in message
