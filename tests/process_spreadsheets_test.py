""" Unit Test Suite that processes Spreadsheet data
"""
import logging
import pytest
from llm_test_lib import LLMTestLib

logger = logging.getLogger(__name__)

def pytest_generate_tests(metafunc):
    if 'test_input' in metafunc.fixturenames:
        llm_test_lib = LLMTestLib()

        # Create test list
        test_list = llm_test_lib.process_test_suites()
        
        # Generate test cases based on the test_data list
        metafunc.parametrize('test_input', test_list)

def test_spreadsheets(test_input):
    llm_test_lib = LLMTestLib()

    messages = []

    # append system prompt
    if test_input.system_prompt is not None and len(test_input.system_prompt) > 0:
        logger.info("System Prompt: %s", test_input.system_prompt)
        messages.append({ "role": "system", "content": test_input.system_prompt })

    # append all the user messages
    msg_counter = 0
    msg_length = len(test_input.messages)
    last_expected_response = None
    for message in test_input.messages:
        msg_counter += 1
        is_last = msg_counter >= msg_length
    
        if is_last:
            last_expected_response = message.expected_response
            messages.append({ "role": "user", "content": message.user_message })
        else:
            messages.append({ "role": "user", "content": message.user_message })
            messages.append({ "role": "assistant", "content": message.expected_response })

    completion = llm_test_lib.invoke_chat_completion_w_messages(messages)
    response = llm_test_lib.get_most_recent(completion)
    assert(llm_test_lib.is_similar(last_expected_response, response, test_input.min_similarity))
