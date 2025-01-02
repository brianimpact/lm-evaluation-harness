from typing import Any, List, Tuple

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

eval_logger = utils.eval_logger


def claude_chat(
        client,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: List[str],
        **kwargs: Any,
) -> str:
    """Wrapper function around the Bedrock chat completion API client with exponential back-off
    in case of RateLimitError.
    params:
        client: anthropic.Anthropic client
        model: str
            anthropic model e.g. 'claude-3-5-sonnet-20241022'
        prompt: str
            Prompt to feed to the model
        max_tokens: int
            Maximum number of tokens to sample from the model
        stop: List[str]
            List of stop sequences
        kwargs: Any
            Additional model_args to pass to the API client
    """

    def messages():
        #    # Set up the base message parameters
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        # Add system message if provided in kwargs
        if "system" in kwargs:
            message_params["system"] = kwargs.pop("system")
        # Make the API call
        response = client.messages.create(**message_params)

        # Return the text content from the response
        return response.content[0].text

    return messages()


@register_model("claude_chat")
class ClaudeChatLM(LM):
    def __init__(
            self,
            model: str,
            temperature: float = 0,
            max_tokens: int = 2048,
            **kwargs,  # top_p, top_k, system etc.
    ) -> None:
        """Anthropic API wrapper.
        :param model: str
            Bedrock model e.g. 'claude-3-5-sonnet-20241022'
        :param max_tokens: int
            Maximum number of tokens to sample from the model
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import anthropic
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install anthropic`",
            )

        self.model = model
        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = None
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        # Not sure but anthropic.HUMAN_PROMPT ?
        raise NotImplementedError("Tokenizer not available.")

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm):
            try:
                inp = request[0]
                request_args = request[1]
                # generation_kwargs
                until = request_args.get("until")
                max_gen_toks = request_args.get("max_gen_toks", self.max_length)
                response = claude_chat(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens=max_gen_toks,
                    temperature=self.temperature,

                    stop=until,  # type: ignore
                    **self.kwargs,
                )
                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)

            except Exception as error:  # 우선 모든 에러 처리 ㅎㅎ..
                eval_logger.critical(f"Anthropic error: {error}")
                break


        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")
