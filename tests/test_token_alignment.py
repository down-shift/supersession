import pytest
from src.data.token_validation import continuation_token_id
from src.experiments.patching import assert_aligned

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
