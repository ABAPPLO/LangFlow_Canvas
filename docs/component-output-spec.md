# Component Output Specification

## Output Types

Langflow uses 4 primary output types. Connections between components require **strict type matching** (case-sensitive, no wildcards).

| Type | Alias | Description | Common Scenarios |
|------|-------|-------------|------------------|
| `Message` | - | Text/chat message | LLM responses, text I/O, prompts |
| `Data` | `JSON` | Structured key-value data | API results, search data, file metadata |
| `DataFrame` | `Table` | Tabular data (rows × columns) | Search results, batch processing, vector retrieval |
| `Tool` | - | LangChain tool object | Search tools, API tools |

Type aliases are equivalent: `Data` ≡ `JSON`, `DataFrame` ≡ `Table`.

## Compatibility Matrix

| Output \ Input | Data/JSON | DataFrame/Table | Message | LanguageModel | Embeddings |
|---|---|---|---|---|---|
| **Data / JSON** | YES | NO | NO | NO | NO |
| **DataFrame / Table** | NO | YES | NO | NO | NO |
| **Message** | NO | NO | YES | NO | NO |
| **LanguageModel** | NO | NO | NO | YES | NO |
| **Embeddings** | NO | NO | NO | NO | YES |

Cross-type connections are NOT allowed. Use converter components as intermediaries:
- `Data` → `Message`: use "Data to Message" component
- `DataFrame` → `Message`: use "Parse DataFrame" component
- `Message` → `DataFrame`: use "Type Convert" component

## Output Declaration Rules

### Text output → use `Message`
```python
from lfx.schema.message import Message

outputs = [
    Output(display_name="Result", name="result", method="process"),
]

def process(self) -> Message:
    return Message(text="hello")
```

### Structured data → use `Data`
```python
from lfx.schema.data import Data

outputs = [
    Output(display_name="Data", name="data", method="process", types=["Data"]),
]

def process(self) -> Data:
    return Data(data={"key": "value"})
```

### Tabular data → use `DataFrame`
```python
from lfx.schema.dataframe import DataFrame

outputs = [
    Output(display_name="Table", name="table", method="process", types=["DataFrame"]),
]

def process(self) -> DataFrame:
    return DataFrame([{"name": "Alice", "age": 25}])
```

### Multiple output types → provide multiple output ports
```python
outputs = [
    Output(display_name="Data", name="data", method="get_data", types=["Data"]),
    Output(display_name="Message", name="message", method="get_message", types=["Message"]),
    Output(display_name="Table", name="table", method="get_table", types=["DataFrame"]),
]
```

## Input Type Reference

| Input Class | `input_types` | Accepts Connections |
|---|---|---|
| `HandleInput` | User-specified | Custom (via `input_types` param) |
| `JSONInput` / `DataInput` | `["Data", "JSON"]` | Data/JSON outputs |
| `DataFrameInput` | `["DataFrame", "Table"]` | DataFrame/Table outputs |
| `TableInput` | `["DataFrame", "Table"]` | DataFrame/Table outputs |
| `MessageInput` | `["Message"]` | Message outputs |
| `MessageTextInput` | `["Message"]` | Message outputs (extracts text) |
| `ModelInput` | `["LanguageModel"]` or `["Embeddings"]` | Model instances |
| `StrInput` | `[]` | No connections (text only) |
| `IntInput` / `FloatInput` / `BoolInput` | `[]` | No connections |
| `SecretStrInput` | `[]` | No connections |
| `PromptInput` | `[]` | No connections |

## Common Pitfalls

| Scenario | Problem | Solution |
|---|---|---|
| LLM output → table processing | `Message` ≠ `DataFrame` | Add Type Convert component |
| Data output → text component | `Data` ≠ `Message` | Add "Data to Message" component |
| Search results → LLM input | `DataFrame` ≠ `Message` | Use Parse Data or Select Data |
| Returning raw `str` | No handle, can't connect | Return `Message(text=...)` instead |

## Loop Components

Loop components (`Loop`, `Field Loop`) use special output flags:
- `allows_loop=True`: allows cycle connections
- `loop_types=["Message"]`: what types the feedback input accepts
- `group_outputs=True`: prevents dropdown selection in UI

## Dynamic Outputs

Components can generate output ports dynamically via `update_outputs()`:

```python
def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
    if field_name == "my_config_field":
        frontend_node["outputs"] = [
            Output(display_name="Dynamic", name="dynamic_1", method="process", types=["Data"]),
        ]
    return frontend_node
```

Trigger with `real_time_refresh=True` on the controlling input field.
