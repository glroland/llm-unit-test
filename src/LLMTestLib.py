""" Library for Unit Testing Expected Prompt Responses from LLMs
"""
import os
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch import tensor

logger = logging.getLogger(__name__)

class LLMTestLib:
    """ Library for unit testing the responses from LLMs """
    ENV_BASE_URL = "BASE_URL"
    ENV_TOKEN = "TOKEN"
    ENV_MODEL = "MODEL"
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_TOKEN = "none"
    DEFAULT_MODEL = None

    base_url = None
    token = None
    model = None
    similarity_threshold = 0.8

    def __init__(self,
                 base_url = None,
                 token = None,
                 model = None):
        """ Default Constructor
        
            base_url - inference service api endpoint
            token - api token
            model - model to execute prompts against
        """
        # Set Base URL
        if base_url is not None:
            self.base_url = base_url
        else:
            self.base_url = self.get_config_value(self.ENV_BASE_URL,
                                                  self.DEFAULT_BASE_URL)

        # Set Token
        if token is not None:
            self.token = token
        else:
            self.token = self.get_config_value(self.ENV_TOKEN,
                                               self.DEFAULT_TOKEN)

        # Set Model
        if model is not None:
            self.model = model
        else:
            self.model = self.get_config_value(self.ENV_MODEL,
                                               self.DEFAULT_MODEL)

    def get_config_value(self, env_name, default_value):
        """ Get a config value from an environment variable.
        
            env_name - environment variable name
            default_value - if default value is not set
        """
        # validate arguments
        if env_name is None or len(env_name) == 0:
            msg = "get_config_value() - env_name is empty"
            logger.error(msg)
            raise ValueError(msg)

        # get key value if exists
        if env_name in os.environ:
            value = os.environ[env_name]
            logger.info("get_config_value() - key was set.  K=%s V=%s", env_name, value)
            return value

        # use default
        logger.warning("get_config_value() - key not set.  using default.  K=%s", env_name)
        return default_value

    def invoke_chat_completion_w_messages(self, messages):
        """ Invoke the OpenAI Chat API with the provided set of messages.
        
            messages - messages
        """
        client = OpenAI()
        client.api_key = self.token
        client.base_url = self.base_url

        logger.info("Invoking Chat Completions with Messages.  %s", messages)

        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        return completion

    def invoke_chat_completion(self, system_prompt, user_message):
        """ Invoke the OpenAI Chat API with the provided prompts.
        
            system_prompt - system prompt
            user_message - user message
        """
        messages = [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]

        return self.invoke_chat_completion_w_messages(messages)

    def invoke_chat_completion_simple(self, user_message):
        """ Invoke the OpenAI Chat API with the provided prompts.
        
            user_message - user message
        """
        messages = [
            { "role": "user", "content": user_message }
        ]

        return self.invoke_chat_completion_w_messages(messages)

    def get_most_recent(self, completion):
        """ Extract the most recent prompt response from the complete chat completion object.
        
            completion - chat completion object
        """
        return completion.choices[0].message.content

    def get_embeddings_for_str(self, str_list):
        """ Creates embeddings for the provided list of strings.
        
            str_list - list of strings to convert
        """
        if str_list is None or len(str_list) == 0:
            msg = "get_embeddings_for_str() was passed an empty list of strings"
            logger.error(msg)
            raise ValueError(msg)
        
        sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return sentence_transformer.encode(str_list)

    def get_cosign_similarity(self, embedding1, embedding2):
        """ Calculates the distance between two embeddings.
        
            embedding1 - embedding 1
            embedding2 - embedding 2
        """
        embedding1_tensor = tensor(embedding1)
        embedding2_tensor = tensor(embedding2)
        return cos_sim(embedding1_tensor, embedding2_tensor).item()

    def is_similar(self, str1, str2):
        """ Determine if the two strings are sufficiently similar. 
        
            str1 - string 1
            str2 - string 2
        """
        # validate arguments
        if str1 is None or len(str1) == 0:
            msg = "is_similar() - provided str1 is empty"
            logger.error(msg)
            raise ValueError(msg)
        if str2 is None or len(str2) == 0:
            msg = "is_similar() - provided str2 is empty"
            logger.error(msg)
            raise ValueError(msg)

        # Calculate distance
        embeddings = self.get_embeddings_for_str([ str1, str2 ])
        similarity = self.get_cosign_similarity(embeddings[0], embeddings[1])
        logger.info("Similarity Results.  CS=%s  T=%s  S1=%s  S2=%s",
                    similarity, self.similarity_threshold, str1, str2)

        return similarity >= self.similarity_threshold


def uppercase_decorator(function):
    def wrapper():
        func = function()
        make_uppercase = func.upper()
        return make_uppercase

    return wrapper

def a_decorator_passing_arbitrary_arguments(function_to_decorate):
    def a_wrapper_accepting_arbitrary_arguments(*args,**kwargs):
        print('The positional arguments are', args)
        print('The keyword arguments are', kwargs)
        function_to_decorate(*args)
    return a_wrapper_accepting_arbitrary_arguments

@a_decorator_passing_arbitrary_arguments
def function_with_no_argument():
    print("No arguments here.")

function_with_no_argument()


def decorator_maker_with_arguments(decorator_arg1, decorator_arg2, decorator_arg3):
    def decorator(func):
        def wrapper(function_arg1, function_arg2, function_arg3) :
            "This is the wrapper function"
            print("The wrapper can access all the variables\n"
                  "\t- from the decorator maker: {0} {1} {2}\n"
                  "\t- from the function call: {3} {4} {5}\n"
                  "and pass them to the decorated function"
                  .format(decorator_arg1, decorator_arg2,decorator_arg3,
                          function_arg1, function_arg2,function_arg3))
            return func(function_arg1, function_arg2,function_arg3)

        return wrapper

    return decorator

pandas = "Pandas"
@decorator_maker_with_arguments(pandas, "Numpy","Scikit-learn")
def decorated_function_with_arguments(function_arg1, function_arg2,function_arg3):
    print("This is the decorated function and it only knows about its arguments: {0}"
           " {1}" " {2}".format(function_arg1, function_arg2,function_arg3))

decorated_function_with_arguments(pandas, "Science", "Tools")
