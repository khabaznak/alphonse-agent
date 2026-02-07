from __future__ import annotations

from alphonse.agent.cognition.message_map_llm import dissect_message


class StubLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self.payload


def test_json_salvage_from_text_block() -> None:
    llm = StubLLM(
        """noise before
{
  "language":"en",
  "social":{"is_greeting":true,"is_farewell":false,"is_thanks":false,"text":"hi"},
  "actions":[],
  "entities":[],
  "constraints":{"times":[],"numbers":[],"locations":[]},
  "questions":[],
  "commands":[],
  "raw_intent_hint":"social_only",
  "confidence":"high",
}
noise after"""
    )
    result = dissect_message("Hi", llm_client=llm)
    assert result.parse_ok is True
    assert result.message_map.social.is_greeting is True


def test_invalid_json_fallback_map() -> None:
    llm = StubLLM("not json")
    result = dissect_message("Hi", llm_client=llm)
    assert result.parse_ok is False
    assert result.message_map.raw_intent_hint == "other"
    assert result.message_map.confidence == "low"
    assert result.message_map.social.is_greeting is True


def test_nested_questions_and_commands_inside_constraints_are_supported() -> None:
    llm = StubLLM(
        """
        {
          "language":"en",
          "social":{"is_greeting":false,"is_farewell":false,"is_thanks":false,"text":null},
          "actions":[],
          "entities":[],
          "constraints":{"times":[],"numbers":[],"locations":[],"questions":["what can you do?"],"commands":["/approve"]},
          "raw_intent_hint":"mixed",
          "confidence":"medium"
        }
        """
    )
    result = dissect_message("what can you do", llm_client=llm)
    assert result.parse_ok is True
    assert result.message_map.questions == ["what can you do?"]
    assert result.message_map.commands == ["/approve"]
