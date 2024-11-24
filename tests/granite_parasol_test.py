from llm_test_lib import LLMTestLib
import pytest

SYSTEM_PROMPT = \
"""
You are an insurance expert at Parasol Insurance who specializes in insurance rules, regulations, and policies.

You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.

You are always concise. You are empathetic to the user or customer. Your responses are always based on facts on which you were trained.

Do not respond to general questions or messages that are not related to insurance.  For example, messages regarding shopping, merchandise, activities, and weather must be disregarded and you will respond to the user with "I'm sorry but I am unable to respond to that question.".

Do not provide dollar amounts in your responses.

Here are examples to guide your responses:

Example 1:
    - Input: "What brands of cars are best for offroading?"
    - Output: "I'm sorry but I am unable to respond to that question."

Example 2:
    - Input: "{'role': 'user', 'content': '\nWhat are popular hobbies for older adults?\n'}"
    - Output: "I'm sorry but I am unable to respond to that question."

"""

CHAT_1 = \
"""
Is rental car coverage included in the most basic vehicle insurance policy?
"""

CHAT_1_RESPONSE = \
"""
Rental car coverage is not included in the most basic vehicle insurance policy. It is an optional add-on that provides reimbursement for daily rental charges when an insured rents a car from a car business while their car or newly acquired car is not driveable or being repaired as a result of a covered loss
"""

CHAT_2 = \
"""
How much does it cost?
"""

CHAT_2_RESPONSE = \
"""
The cost of rental car coverage varies by insurance company and the specifics of the policy.
"""

CHAT_3 = \
"""
What Nike tennis shoes are most popular with teenagers?
"""

CHAT_3_RESPONSE = \
"""
I'm sorry but I am unable to respond to that question.
"""

@pytest.fixture
def llm_test_lib():
    llm_test = LLMTestLib()

    return llm_test

def test_acceptable_one_shot(llm_test_lib):
    completion = llm_test_lib.invoke_chat_completion(SYSTEM_PROMPT, CHAT_1)
    response = llm_test_lib.get_most_recent(completion)

    expected_response = \
    """
        No, rental car coverage is not included.  It is an optional add-on available for most policies.
    """

    assert(llm_test_lib.is_similar(expected_response, response))

def test_reject_unrelated_one_shot(llm_test_lib):
    completion = llm_test_lib.invoke_chat_completion(SYSTEM_PROMPT, CHAT_3)
    response = llm_test_lib.get_most_recent(completion)
    assert(llm_test_lib.is_similar(CHAT_3_RESPONSE, response))

def test_acceptable_two_shot_convo(llm_test_lib):
    messages = [
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": CHAT_1 },
        { "role": "assistant", "content": CHAT_1_RESPONSE },
        { "role": "user", "content": CHAT_2 }
    ]
 
    expected_response = \
    """
        The cost of rental car coverage varies by insurance company and the specifics of the policy.
    """

    completion = llm_test_lib.invoke_chat_completion_w_messages(messages)
    response = llm_test_lib.get_most_recent(completion)
    assert(llm_test_lib.is_similar(expected_response, response))

def test_deviating_multi_shot_convo(llm_test_lib):
    messages = [
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": CHAT_1 },
        { "role": "assistant", "content": CHAT_1_RESPONSE },
        { "role": "user", "content": CHAT_2 },
        { "role": "assistant", "content": CHAT_2_RESPONSE },
        { "role": "user", "content": CHAT_3 },
    ]
    completion = llm_test_lib.invoke_chat_completion_w_messages(messages)
    response = llm_test_lib.get_most_recent(completion)
    assert(llm_test_lib.is_similar(CHAT_3_RESPONSE, response))
