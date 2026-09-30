import time
from typing import Any, Dict, Optional

import hummingbot.connector.derivative.gate_io_perpetual.gate_io_perpetual_constants as CONSTANTS
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.rest_post_processors import RESTPostProcessorBase
from hummingbot.core.web_assistant.rest_pre_processors import RESTPreProcessorBase
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory


def public_rest_url(endpoint: str, domain: str = CONSTANTS.DEFAULT_DOMAIN) -> str:
    """
    Creates a full URL for provided public REST endpoint

    :param endpoint: a public REST endpoint
    :param domain: unused

    :return: the full URL to the endpoint
    """
    if CONSTANTS.REST_URL[-1] != "/" and endpoint[0] != "/":
        endpoint = "/" + endpoint
    return CONSTANTS.REST_URL + endpoint


def private_rest_url(endpoint: str, domain: str = CONSTANTS.DEFAULT_DOMAIN) -> str:
    return public_rest_url(endpoint, domain)


def build_api_factory(
        throttler: Optional[AsyncThrottler] = None,
        auth: Optional[AuthBase] = None,
        guard_provider=None) -> WebAssistantsFactory:
    throttler = throttler or create_throttler()
    api_factory = WebAssistantsFactory(
        throttler=throttler,
        auth=auth,
        rest_pre_processors=[GateGuardPreProcessor(guard_provider)] if guard_provider else [],
        rest_post_processors=[GateGuardPostProcessor(guard_provider)] if guard_provider else [])
    return api_factory


class GateGuardPreProcessor(RESTPreProcessorBase):
    def __init__(self, provider):
        self.provider = provider

    async def pre_process(self, request):
        guard = self.provider()
        if guard:
            guard.before_request(request.method.value, request.url, request.data)
        return request


class GateGuardPostProcessor(RESTPostProcessorBase):
    def __init__(self, provider):
        self.provider = provider

    async def post_process(self, response):
        guard = self.provider()
        if guard:
            guard.after_response(response.method.value, response.url, response.status, response.headers)
        return response


def create_throttler() -> AsyncThrottler:
    return AsyncThrottler(CONSTANTS.RATE_LIMITS)


async def get_current_server_time(
        throttler: Optional[AsyncThrottler] = None,
        domain: str = CONSTANTS.DEFAULT_DOMAIN,
) -> float:
    return time.time()


def is_exchange_information_valid(exchange_info: Dict[str, Any]) -> bool:
    """
    Verifies if a trading pair is enabled to operate with based on its exchange information
    :param exchange_info: the exchange information for a trading pair
    :return: True if the trading pair is enabled, False otherwise
    """
    return exchange_info.get("in_delisting", None) is False
