import unittest
from pydantic import BaseModel
from prototype_phd.langchain_utils import (
    map_characters_to_token_indices,
    extract_json_data,
    add_logprobs,
    add_top_logprobs,
    ChatCompletionTokenLogprob,
)

# Updated dummy classes:

class DummyMessage:
    def __init__(self, content: str):
        self.content = content

class DummyChoice:
    def __init__(self, content: str, additional_kwargs: dict = None, response_metadata: dict = None):
        self.message = DummyMessage(content)
        self.additional_kwargs = additional_kwargs or {}
        self.response_metadata = response_metadata or {}

    @property
    def logprobs(self):
        # Return the logprobs from response_metadata if available.
        return self.response_metadata.get("logprobs", None)

class DummyChatResponse(BaseModel):
    choices: list

class TestLangchainUtils(unittest.TestCase):
    def test_map_characters_to_token_indices(self):
        tokens = [
            ChatCompletionTokenLogprob(token="abc", logprob=1.0),
            ChatCompletionTokenLogprob(token="de", logprob=2.0)
        ]
        expected = [0,0,0,1,1]
        result = map_characters_to_token_indices(tokens)
        self.assertEqual(result, expected)

    def test_extract_json_data_string(self):
        # Use a simple JSON string value.
        json_string = '"hello"'
        token = ChatCompletionTokenLogprob(token=json_string, logprob=3.0)
        tokens = [token]
        token_indices = [0] * len(json_string)
        # Just verify that a value is returned (its exact nature depends on Lark positions).
        result = extract_json_data(json_string, tokens, token_indices)
        # Expect the string to be parsed and transformed (here the transformer returns a float for string)
        self.assertIsInstance(result, float)

    def test_add_logprobs_success(self):
        text = '"text"'
        # Split the text into two tokens.
        half = len(text) // 2
        token1_text, token2_text = text[:half], text[half:]
        token1 = ChatCompletionTokenLogprob(token=token1_text, logprob=0.5)
        token2 = ChatCompletionTokenLogprob(token=token2_text, logprob=0.5)
        response_metadata = {"logprobs": {"content": [token1, token2]}}
        choice = DummyChoice(content=text, response_metadata=response_metadata)
        dummy_response = DummyChatResponse(choices=[choice])
        result = add_logprobs(dummy_response)
        self.assertEqual(result.value, dummy_response)
        self.assertEqual(len(result.log_probs), 1)
        # The cumulative logprob should equal 1.0 (0.5+0.5).
        self.assertIsInstance(result.log_probs[0], float)
        self.assertAlmostEqual(result.log_probs[0], 1.0)

    def test_add_logprobs_failure(self):
        # Missing logprobs in response_metadata.
        choice = DummyChoice(content='"text"', response_metadata={})
        dummy_response = DummyChatResponse(choices=[choice])
        with self.assertRaises(AttributeError):
            add_logprobs(dummy_response)

    def test_add_top_logprobs_success(self):
        token = ChatCompletionTokenLogprob(token='"text"', logprob=1.5, top_logprobs={'"text"': 1.5})
        response_metadata = {"logprobs": {"top_logprobs": [token]}}
        choice = DummyChoice(content='"text"', response_metadata=response_metadata)
        dummy_response = DummyChatResponse(choices=[choice])
        result = add_top_logprobs(dummy_response)
        self.assertEqual(result.value, dummy_response)
        self.assertEqual(len(result.top_log_probs), 1)

    def test_add_top_logprobs_failure(self):
        choice = DummyChoice(content='"text"', response_metadata={})
        dummy_response = DummyChatResponse(choices=[choice])
        with self.assertRaises(AttributeError):
            add_top_logprobs(dummy_response)

    def test_chat_completion_example_logprobs(self):
        # Simulated chat completion content from console output.
        content = '{\n  "setup": "Why couldn\'t the bicycle stand up by itself?",\n  "punchline": "Because it was two tired!"\n}'
        # Split content into two tokens.
        half = len(content) // 2
        token1_text, token2_text = content[:half], content[half:]
        token1 = ChatCompletionTokenLogprob(token=token1_text, logprob=-0.5, top_logprobs={token1_text: -0.5})
        token2 = ChatCompletionTokenLogprob(token=token2_text, logprob=-0.5, top_logprobs={token2_text: -0.5})
        response_metadata = {"logprobs": {"content": [token1, token2]}}
        choice = DummyChoice(content=content, response_metadata=response_metadata)
        dummy_response = DummyChatResponse(choices=[choice])
        result = add_logprobs(dummy_response)
        self.assertEqual(result.value, dummy_response)
        self.assertEqual(len(result.log_probs), 1)
        # The JSON parser returns a dictionary; for the "setup" key, expect a float cumulative logprob.
        self.assertIn("setup", result.log_probs[0])
        self.assertIsInstance(result.log_probs[0]["setup"], float)
        self.assertAlmostEqual(result.log_probs[0]["setup"], -1.0)
        
    
    def test_chat_completion_example_top_logprobs(self):
        # Simulated chat completion content from console output.
        content = '{\n  "setup": "Why couldn\'t the bicycle stand up by itself?",\n  "punchline": "Because it was two tired!"\n}'
        # Split content into two tokens to simulate a realistic scenario.
        half = len(content) // 2
        token1_text = content[:half]
        token2_text = content[half:]
        token1 = ChatCompletionTokenLogprob(token=token1_text, logprob=-0.5, top_logprobs={token1_text: -0.5})
        token2 = ChatCompletionTokenLogprob(token=token2_text, logprob=-0.5, top_logprobs={token2_text: -0.5})
        response_metadata = {"logprobs": {"top_logprobs": [token1, token2]}}
        choice = DummyChoice(content=content, response_metadata=response_metadata)
        dummy_response = DummyChatResponse(choices=[choice])
        result = add_top_logprobs(dummy_response)
        self.assertEqual(result.value, dummy_response)
        self.assertEqual(len(result.top_log_probs), 1)
        # Now, for example, the "setup" field should have alternatives coming from both tokens.
        # Assert that result.top_log_probs[0]['setup'] is a list.
        self.assertIsInstance(result.top_log_probs[0]['setup'], list)

if __name__ == '__main__':
    unittest.main()
