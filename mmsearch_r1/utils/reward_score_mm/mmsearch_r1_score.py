# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string


# adapted from search-r1
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    '''
    prediction: string
    golden_answers: list or string, support multi candidate answers
    '''
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    exactly_match = False
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            exactly_match = True
            break
    return exactly_match


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(prediction):
    """从最后一轮回复中抽取 <answer>...</answer>"""
    match = list(re.finditer(r'<answer>(.*?)</answer>', prediction, re.DOTALL))
    return None if not match else match[-1].group(1).strip()


# --------------------------- 输出格式校验 --------------------------- #

def is_valid_direct_answer(resp, pattern):
    """
    合法直接回答格式:
      <think>...</think><answer>...</answer>
    无任何搜索调用。
    """
    if not re.match(pattern, resp, re.DOTALL):
        return False
    if resp.count('<think>') != 1 or resp.count('</think>') != 1:
        return False
    if resp.count('<answer>') != 1 or resp.count('</answer>') != 1:
        return False
    # 禁止出现任何搜索标签
    if '<text_search>' in resp or '</text_search>' in resp:
        return False
    return True


def is_valid_text_search(resp, pattern):
    """
    合法文本检索格式:
      <think>...</think><text_search>...</text_search>
    不允许出现 <answer> 标签。
    """
    if not re.match(pattern, resp, re.DOTALL):
        return False
    if resp.count('<think>') != 1 or resp.count('</think>') != 1:
        return False
    if resp.count('<text_search>') != 1 or resp.count('</text_search>') != 1:
        return False
    if '<answer>' in resp or '</answer>' in resp:
        return False
    return True


# --------------------------- Format + Search 评估 --------------------------- #

def format_reward(responses):
    """
    只保留两种合规形式：
      1-turn:  直接回答
      2-turn:  文本检索 → 回答
    """
    direct_pattern = r'^<think>.*</think>.*<answer>.*</answer>$'
    text_search_pattern = r'^<think>.*</think>.*<text_search>.*</text_search>$'

    rounds = len(responses)
    fmt_score, search_cnt = 0, 0

    if rounds == 1:
        r1 = responses[0].strip()
        if is_valid_direct_answer(r1, direct_pattern):
            fmt_score = 1
        # count search (仅统计第一次出现即可)
        if '<text_search>' in r1 and '</text_search>' in r1:
            search_cnt = 1

    elif rounds == 2:
        r1, r2 = responses[0].strip(), responses[1].strip()
        if is_valid_text_search(r1, text_search_pattern) and is_valid_direct_answer(r2, direct_pattern):
            fmt_score = 1
        if '<text_search>' in r1 and '</text_search>' in r1:
            search_cnt = 1
            

    else:
        raise ValueError(f"Unsupported number of turns: {rounds}. Only 1 or 2 are allowed now.")

    return fmt_score, search_cnt


# --------------------------- 主奖励函数 --------------------------- #

def compute_score(prediction: list, ground_truth: list, extra_info=None):
    # Exactly Match Scorer
    search_penalty, format_penalty = 0.1, 0.1
    reward_mode = 'EM'
    if extra_info is not None and 'search_penalty' in extra_info:
        search_penalty = extra_info.get('search_penalty', 0.1)
    if extra_info is not None and 'format_penalty' in extra_info:
        format_penalty = extra_info.get('format_penalty', 0.1)
    if extra_info is not None and 'reward_mode' in extra_info:
        reward_mode = extra_info.get('reward_mode', 'EM')
        assert reward_mode in ['EM', 'SubEM'], f'reward mode {reward_mode} passed in extra_info but not recognized'

    # Extract Answer
    assert len(prediction) > 0, "[Error Occured] Model Responses are empty!"
    answer = extract_solution(prediction=prediction[-1])

    score = 0
    # Correctness Check: EM/SubEM
    if answer is not None:
        if reward_mode == "EM" and em_check(answer, ground_truth):
            score = 1
        elif reward_mode == 'SubEM' and subem_check(answer, ground_truth):
            correct = 1

    # Format Check
    format_score, search_count = format_reward(prediction)

    # Search Penalty, 0.99 is added here because we only want to punish correct answers
    if search_count > 0 and score > 0.99:
        use_search_count_penalty = extra_info.get('use_search_count_penalty', False)
        if use_search_count_penalty:
            # penalty w/ search count
            for _ in range(search_count):
                score *= 1 - search_penalty
        else:
            # penalty w/o search count
            score *= 1 - search_penalty  # no penalty when not correct

    # Weighted Score: (1-FP) * Score + FP * Format_Score
    return (1 - format_penalty) * score + format_penalty * format_score
