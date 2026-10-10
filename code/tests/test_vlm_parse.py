"""Offline checks of the VLM client's answer parsing: yes/no logprobs, the three-way template (partly counts as no),
count answers (digit distribution, ten or more) and box-list counting; _post is stubbed, no server needed."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redag.vlm import VLM, digit_dist, parse_count  # noqa: E402


def fake_post(top):
    def _post(self, body):
        return {"choices": [{"message": {"content": top[0][0]}, "logprobs": {"content": [{"top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top]}]}}]}
    return _post


def test_parse():
    v = VLM("http://127.0.0.1:1/v1", "stub", require_gate=False)
    VLM._post = fake_post([("yes", 0.6), ("partly", 0.3), ("no", 0.1)])
    p, z, ok = v._yes_prob_logprob([], "q", three_way=False)
    assert ok and abs(p - 0.6 / 0.7) < 1e-6, p
    p3, z3, ok = v._yes_prob_logprob([], "q", three_way=True)
    assert ok and abs(p3 - 0.6) < 1e-6, p3
    VLM._post = fake_post([("Yes", 0.9), ("No", 0.1)])
    p, _, _ = v._yes_prob_logprob([], "q", three_way=True)
    assert abs(p - 0.9) < 1e-6
    VLM._post = fake_post([("maybe", 0.9)])
    p, _, ok = v._yes_prob_logprob([], "q")
    assert not ok and p == 0.5 and v.fmt_fail == 1
    dd = digit_dist([{"token": "1", "logprob": math.log(0.7)}, {"token": "2", "logprob": math.log(0.2)}, {"token": "x", "logprob": math.log(0.1)}])
    assert set(dd) == {1, 2} and abs(dd[1] - 0.7 / 0.9) < 1e-9 and abs(dd[2] - 0.2 / 0.9) < 1e-9, dd
    assert parse_count("There are 12 cats") == 12 and parse_count("two") == 2 and parse_count("none") == 0 and parse_count("") is None
    from PIL import Image

    im = Image.new("RGB", (64, 64), (1, 2, 3))
    VLM._post = fake_post([("2", 0.8), ("1", 0.2)])
    p, z, k = v.count_one(im, "cats", 1)
    assert k == 2 and abs(p - 0.2) < 1e-6, (p, k)
    p, z, k = v.count_one(im, "cats", 2)
    assert abs(p - 0.999) < 1e-6 or abs(p - 1.0) < 1e-3, p
    VLM._post = fake_post([("12", 0.9)])
    p, z, k = v.count_one(im, "dogs", 1)
    assert k == 12 and p <= 0.001, (p, k)


if __name__ == "__main__":
    test_parse()
    print("vlm parse tests ok")
