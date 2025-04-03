from typing import Any, TypeAlias, Optional, Dict, List, Tuple
from lark import Lark, Token, Transformer_NonRecursive, Tree, v_args
from lark.tree import Meta
from pydantic import BaseModel


MISSING_LOGPROBS_MESSAGE = "Missing logprobs in the response for one or more choices."
 
class ChatCompletionTokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[Dict[str, float]] = None
    bytes: Optional[int] = None

ChatCompletion: TypeAlias = Any # a completion object returned by langchain.

PyTree: TypeAlias = Any  # a tree-like structure built out of container-like Python objects.

# Define a grammar for JSON
json_grammar = r"""
    start: value

    ?value: object              #'?' is a Lark convention indicating that the rule can return the value directly instead of creating a separate parse tree node.
          | array
          | string
          | SIGNED_NUMBER -> number    #'-> number' specifies an alias for the rule
          | "true"
          | "false"
          | "null"

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : key ":" value
    key    : ESCAPED_STRING

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


# Transformer that processes the tree and substitutes each atomic value with the cumulative log-probability of its tokens
@v_args(meta=True)
class Extractor(Transformer_NonRecursive):
    def __init__(self, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]):
        super().__init__()
        self.tokens = tokens
        self.token_indices = token_indices

    def _compute_logprob_sum(self, start: int, end: int) -> float:
        token_start = self.token_indices[start]
        token_end = self.token_indices[end - 1]  # adjust for exclusive end_pos
        sum_logporb = sum(self.tokens[i].logprob for i in range(token_start, token_end + 1))
        return sum_logporb

    def number(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def string(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def true(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def false(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def null(self, meta: Meta, children: list[Token]) -> None:
        return None

    def array(self, meta: Meta, children: list[Any]) -> list[float]:
        return children

    def object(self, meta: Meta, children: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in children:
            result[key] = value
        return result

    def pair(self, meta: Meta, children: list[Any]) -> tuple[str, Any]:
        value = children[1]
        key = children[0]
        if isinstance(value, Tree) and not value.children:  # ['b', Tree(Token('RULE', 'value'), [])]
            value = None
        return key, value

    def key(self, meta: Meta, children: list[Token]) -> str:
        return children[0][1:-1]

    def start(self, meta: Meta, children: list[dict[str, Any]]) -> dict[str, Any]:
        return children[0]


class Extractor_top_logprobs(Extractor):
    def __init__(self, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]):
        super().__init__(tokens, token_indices)
    
    
    def _compute_top_alternatives(self, start: int, end: int) -> List[Tuple[str, float]]:
        """
        Compute candidate alternative strings for the substring defined by character positions [start, end].
        
        Returns:
            A list of dictionaries of alternative_token to logprobs, one for each generated token in the substring.
        """
        # Map character positions to token indices.
        token_start = self.token_indices[start]
        token_end = self.token_indices[end - 1]  # adjust for exclusive end_pos
        n_tokens = token_end - token_start + 1

        top_alternatives = []
        for i in range(n_tokens):
            token = self.tokens[token_start + i]
            orig_token = token.token
            # Assume self.tokens[i] has an attribute "top_logprobs"
            alternatives = token.top_logprobs if token.top_logprobs is not None else {}
            # Ensure the original token is included.
            if orig_token not in alternatives:
                alternatives[orig_token] = token.logprob
            top_alternatives.append(alternatives)
        return top_alternatives
    
    @v_args(meta=True)
    def number(self, meta: Meta, children: list[Token]) -> float:
        top_alternatives = self._compute_top_alternatives(meta.start_pos, meta.end_pos)
        return top_alternatives

    @v_args(meta=True)
    def string(self, meta: Meta, children: list[Token]) -> float:
        top_alternatives = self._compute_top_alternatives(meta.start_pos, meta.end_pos)
        return top_alternatives

    @v_args(meta=True)
    def true(self, meta: Meta, children: list[Token]) -> float:
        top_alternatives = self._compute_top_alternatives(meta.start_pos, meta.end_pos)
        return top_alternatives

    @v_args(meta=True)
    def false(self, meta: Meta, children: list[Token]) -> float:
        top_alternatives = self._compute_top_alternatives(meta.start_pos, meta.end_pos)
        return top_alternatives
    
# TODO: Check whether extract_json_data can handle a json_string which is not a valid JSON string.

def extract_json_data(json_string: str, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]) -> PyTree:
    json_parser = Lark(json_grammar, parser="lalr", propagate_positions=True, maybe_placeholders=False)
    tree = json_parser.parse(json_string)
    extractor = Extractor(tokens, token_indices)
    return extractor.transform(tree)

def extract_top_logprobs(json_string: str, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]) -> PyTree:
    json_parser = Lark(json_grammar, parser="lalr", propagate_positions=True, maybe_placeholders=False)
    tree = json_parser.parse(json_string)
    extractor = Extractor_top_logprobs(tokens, token_indices)
    return extractor.transform(tree)


class ChatCompletionWithLogProbs(BaseModel):
    value: ChatCompletion
    log_probs: list[Any]


def map_characters_to_token_indices(extracted_data_token: list[ChatCompletionTokenLogprob]) -> list[int]:
    """
    Maps each character in the JSON string output to its corresponding token index.

    Args:
    extracted_data_token : A list of `TokenLogprob` objects, where each object represents a token and its associated data.

    Returns:
    A list of integers where each position corresponds to a character in the concatenated JSON string,
    and the integer at each position is the index of the token responsible for generating that specific character.
    Example:
        >>> tokens = [ChatCompletionTokenLogprob(token='{'),
                      ChatCompletionTokenLogprob(token='"key1"'),
                      ChatCompletionTokenLogprob(token=': '),
                      ChatCompletionTokenLogprob(token='"value1"'),
                      ChatCompletionTokenLogprob(token='}')]
        >>> map_characters_to_token_indices(tokens)
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    """

    token_indices = []

    for token_idx, token_data in enumerate(extracted_data_token):
        token_text = token_data.token
        token_indices.extend([token_idx] * len(token_text))

    return token_indices


def add_logprobs(chat_completion_response: ChatCompletion) -> ChatCompletionWithLogProbs:
    """
    Adds log probabilities to the chat completion response and returns a
    ChatCompletionWithLogProbs object.

    Args:
        chat_completion_response: The OpenAI chat completion response.

    Returns:
        An object containing:
            - The original chat completion response.
            - A `log_probs` field, structured like the message.content of the response,
              where values are replaced with their respective log-probabilities.
    Raises:
        AttributeError: If any 'choice' in the response does not contain 'logprobs'.

    """

    logprobs_data = []
    for choice in chat_completion_response.choices:
        # Handle both dict and object formats for logprobs
        logprobs_obj = choice.logprobs
        if logprobs_obj is not None:
            if isinstance(logprobs_obj, dict):
                content_value = logprobs_obj.get("content")
            else:
                content_value = getattr(logprobs_obj, "content", None)
        else:
            content_value = None

        if content_value is not None:
            extracted_data = choice.message.content
            logprobs_list = content_value
            token_indices = map_characters_to_token_indices(logprobs_list) if logprobs_list else []
            json_dict = extract_json_data(extracted_data, logprobs_list, token_indices) if extracted_data else {}
            logprobs_data.append(json_dict)
        else:
            raise AttributeError(MISSING_LOGPROBS_MESSAGE)

    chat_completion_with_logprobs = ChatCompletionWithLogProbs(value=chat_completion_response, log_probs=logprobs_data)
    return chat_completion_with_logprobs


class ChatCompletionWithTopLogProbs(BaseModel):
    value: ChatCompletion
    top_log_probs: list[Any]


def add_top_logprobs(chat_completion_response: ChatCompletion) -> ChatCompletionWithTopLogProbs:
    """
    Adds top log probability information to the chat completion response and returns an
    object containing the original response along with a structure parsed from the
    message content in which the values are replaced by the aggregated top_logprobs 
    for the corresponding fields.
    
    It is highly recommended that you only use this when field values are expected
    to be single tokens. The top_logprobs are only available for the tokens that were
    actually generated, and the logprobs are conditional on the earlier generated tokens.
    It is impossible to know the distribution of longer sequences from a single prompt.
    
    Args:
        chat_completion_response: The OpenAI chat completion response.
    
    Returns:
        A ChatCompletionWithTopLogProbs object containing:
            - value: The original chat completion response.
            - top_log_probs: For each choice, a JSON dict where the values are replaced 
              with the aggregated top logprobs (based on character/token mappings).
    
    Raises:
        AttributeError: If any choice in the response does not contain top_logprobs.

    """
    # In general, we can use the logprobs in the following ways:
    # (i) Know how likely the exact generated subsequence of tokens is.
    # (ii) Know the likelihood of choosing different tokens at a given position.
    # (iii) Know the likelihood of the first token of an alternative sequence.

    # Ways to know more with multiple prompts:
    # (i) Ask for single token responses to make choice.
    # (ii) Ask to generate alternative full answers in response to that choice.
    # (iii) Ask for the model to generate alternative choices which are mapping to
    # a single unique token (even perhaps in the same prompt but could be done
    # in a seperate prompt).
    # (iv) Have a shared pool of alternative choices that models can make use of
    # to generate their responses. This allows information to accumulate across
    # agents (allowing a form of collective learning), while still allowing us
    # to know the logprobs associated with a wide and adaptive choice set. These
    # logprobs let us see how a model's choice preferences are changing over time
    # in response to new history or shared knowledge. The key to getting this to
    # work is to let the discrete choices be made first, and only then followed
    # by contributions to a shared set of knowledge.
    top_logprobs_data = []
    for choice in chat_completion_response.choices:
        # Handle both dict and object formats for top_logprobs.
        logprobs_obj = choice.logprobs
        if logprobs_obj is not None:
            if isinstance(logprobs_obj, dict):
                top_value = logprobs_obj.get("top_logprobs")
            else:
                top_value = getattr(logprobs_obj, "top_logprobs", None)
        else:
            top_value = None

        if top_value is not None:
            extracted_data = choice.message.content
            top_logprobs_list = top_value
            token_indices = map_characters_to_token_indices(top_logprobs_list) if top_logprobs_list else []
            json_dict = extract_top_logprobs(extracted_data, top_logprobs_list, token_indices) if extracted_data else {}
            top_logprobs_data.append(json_dict)
        else:
            raise AttributeError("Missing top_logprobs in the response for one or more choices.")

    chat_completion_with_top_logprobs = ChatCompletionWithTopLogProbs(
        value=chat_completion_response,
        top_log_probs=top_logprobs_data
    )
    return chat_completion_with_top_logprobs
