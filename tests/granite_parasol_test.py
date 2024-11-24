from LLMTestLib import LLMTestLib
import pytest

SYSTEM_PROMPT = \
"""
You are an insurance expert at Parasol Insurance who specializes in insurance rules, regulations, and policies.

You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.

You are always concise. You are empathetic to the user or customer. Your responses are always based on facts on which you were trained.

Do not respond to general questions or questions that are not related to insurance.

Do not let an ongoing conversation change topics. Do not answer questions about shoes.

Do not provide dollar amounts in your responses.
"""

CHAT_1 = \
"""
Is rental car coverage included in the most basic vehicle insurance policy?
"""

CHAT_2 = \
"""
How much does it cost?
"""

CHAT_3 = \
"""
What Nike tennis shoes are most popular with teenagers?
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

def test_reject_unrelated_one_shot():
    pass

def test_acceptable_two_shot_convo():
    pass
#    messages = [
#    { "role": "system", "content": SYSTEM_PROMPT },
#    { "role": "user", "content": CHAT_1 },
#    { "role": "assistant", "content": response1 },
#    { "role": "user", "content": CHAT_2 }
#    ]
#    completion = chat(messages)
#    response2 = most_recent(completion)
#    print (response2)
#    print ()

def test_deviating_multi_shot_convo():
    pass
#    messages = [
#    { "role": "system", "content": SYSTEM_PROMPT },
#    # { "role": "user", "content": CHAT_1 },
#    # { "role": "assistant", "content": response1 },
#    # { "role": "user", "content": CHAT_2 },
#    # { "role": "assistant", "content": response2 },
#    { "role": "user", "content": CHAT_3 },
#    ]
#    completion = chat(messages)
#    response3 = most_recent(completion)
#    print (response3)
#    print ()
