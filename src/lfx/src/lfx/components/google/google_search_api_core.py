from langchain_google_community import GoogleSearchAPIWrapper

from lfx.custom.custom_component.component import Component
from lfx.io import IntInput, MultilineInput, Output, SecretStrInput
from lfx.schema.dataframe import DataFrame


class GoogleSearchAPICore(Component):
    display_name = "Google Search API"
    description = "Call Google Search API and return results as a DataFrame."
    icon = "Google"

    inputs = [
        SecretStrInput(
            name="google_api_key",
            display_name="Google API Key",
            required=True,
        ),
        SecretStrInput(
            name="google_cse_id",
            display_name="Google CSE ID",
            required=True,
        ),
        MultilineInput(
            name="input_value",
            display_name="Input",
            tool_mode=True,
        ),
        IntInput(
            name="k",
            display_name="Number of results",
            value=4,
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Results",
            name="results",
            type_=DataFrame,
            method="search_google",
        ),
    ]

    def search_google(self) -> DataFrame:
        """Search Google using the provided query."""
        if not self.google_api_key:
            msg = "Invalid Google API Key"
            raise ValueError(msg)

        if not self.google_cse_id:
            msg = "Invalid Google CSE ID"
            raise ValueError(msg)

        try:
            wrapper = GoogleSearchAPIWrapper(
                google_api_key=self.google_api_key, google_cse_id=self.google_cse_id, k=self.k
            )
            results = wrapper.results(query=self.input_value, num_results=self.k)
        except (ValueError, KeyError) as e:
            msg = f"Invalid Google Search configuration for query '{self.input_value}': {e}"
            raise ValueError(msg) from e
        except ConnectionError as e:
            msg = f"Connection error during Google Search for query '{self.input_value}': {e}"
            raise ConnectionError(msg) from e
        except RuntimeError as e:
            msg = f"Error occurred during Google Search for query '{self.input_value}': {e}"
            raise RuntimeError(msg) from e

        return DataFrame(results)

    def build(self):
        return self.search_google
