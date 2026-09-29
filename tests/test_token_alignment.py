import pytest
from src.data.token_validation import continuation_token_id
from src.experiments.patching import assert_aligned
from src.data.generate import render_example

class ToyTokenizer:
    chat_template=None
    def __call__(self,text,add_special_tokens=False):
        # Deliberately models boundary-sensitive tokenization: "Q amber" is one
        # token, so a bare-word token check would be scientifically invalid.
        tokens=text.split()
        if len(tokens)>=2 and tokens[-2:]==["Q","amber"]: tokens=tokens[:-2]+["Q_amber"]
        return {"input_ids":[abs(hash(x))%100000 for x in tokens]}

def test_single_continuation_checked_against_full_prefix():
    class Simple:
        def __call__(self,text,add_special_tokens=False): return {"input_ids":[{"Q:":1,"amber":2}[x] for x in text.split()]}
    assert continuation_token_id(Simple(),"Q:"," amber")==2

def test_boundary_merge_is_rejected():
    with pytest.raises(ValueError): continuation_token_id(ToyTokenizer(),"Q"," amber")

def test_alignment_equal_and_exact_difference_positions():
    assert_aligned([1,2,3],[1,9,3],[1])
    with pytest.raises(ValueError): assert_aligned([1,2,3],[1,9],[1])
    with pytest.raises(ValueError): assert_aligned([1,2,3],[1,9,3],[2])

def test_chat_renderer_disables_thinking_and_prefills_answer():
    class ChatTokenizer:
        chat_template = "present"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is True
            assert "Respond with only the value" in messages[0]["content"]
            return "<|im_start|>assistant\n"

    example = {
        "family": "symbolic", "variables": ["x", "z"],
        "entities": {"x": "Nora", "z": "Liam"},
        "order": ["O_x", "O_z", "C_x", "C_z"], "query": "x",
        "template_id": 0, "format_id": 0, "old_x": "amber", "old_z": "birch",
        "current_x": "coral", "current_z": "denim", "direct": False,
    }
    assert render_example(example, ChatTokenizer()).endswith("Answer:")

def test_non_chat_renderer_has_explicit_answer_prefix():
    example = {
        "family": "symbolic", "variables": ["x", "z"],
        "entities": {"x": "Nora", "z": "Liam"},
        "order": ["O_x", "O_z", "C_x", "C_z"], "query": "x",
        "template_id": 0, "format_id": 0, "old_x": "amber", "old_z": "birch",
        "current_x": "coral", "current_z": "denim", "direct": False,
    }
    assert render_example(example, chat=False).endswith("Answer:")
