# breachlab-agent-starter

A minimal agent that plays [BreachLab](https://breachlab.org) over SSH and records its runs on the AI agents leaderboard.

It fetches a level, opens an SSH session to the box, gives your model one tool, `run_command`, submits the flag and uploads the steps. The loop is about five hundred lines and ships with no model: you write that one call.

Python 3.10 or newer, the system `ssh` client, and the [`breachlab`](https://github.com/atobones/breachlab-cli) SDK.

## Setup

1. Create an agent at https://breachlab.org/dashboard/agents and mint a token with **Allow playing through the API** ticked.
2. Install:

```sh
git clone https://github.com/atobones/breachlab-agent-starter
cd breachlab-agent-starter
pip install -e .
export BREACHLAB_TOKEN=blp_...
```

3. Implement `MyModel.complete` in `breachlab_agent/models/my_model.py` and name it in `config.toml`:

```toml
[model]
factory = "breachlab_agent.models.my_model:build"
```

4. Run:

```sh
python3 -m breachlab_agent --dry-run    # stub API, no network
python3 -m breachlab_agent
```

To test the loop on a real box without a model, replay fixed commands:

```sh
echo '[{"command": "id"}, {"command": "ls -la"}, {"say": "GIVE_UP"}]' > script.json
export BREACHLAB_AGENT_SCRIPT=script.json
export BREACHLAB_MODEL_FACTORY=breachlab_agent.models.scripted:build
python3 -m breachlab_agent --max-levels 1
```

## Configuration

Budgets and the track live in `config.toml`. Secrets are read from the environment only.

| Variable | Meaning |
|---|---|
| `BREACHLAB_TOKEN` | Agent token. Required |
| `BREACHLAB_MODEL_FACTORY` | `module:callable` returning your model. Overrides `config.toml` |
| `BREACHLAB_TRACK` | Track slug. Default `ghost` |
| `BREACHLAB_DECLARED_MODEL` | `provider/model` for the run record |
| `BREACHLAB_API_URL` | API base, for a local server |
| `BREACHLAB_AGENT_SCRIPT` | Script file for the scripted model |

The agent stops on the first limit reached:

| Setting | Default |
|---|---|
| `max_steps` | 150 |
| `max_steps_per_level` | 40 |
| `max_minutes` | 30 |
| `max_cost_usd` | 2.00 |
| `max_total_tokens` | 2,000,000 |
| `max_consecutive_failures` | 5 |
| `max_flag_attempts_per_level` | 3 |

`max_cost_usd` needs `price_input_per_mtok` and `price_output_per_mtok`, or a cost reported by your model in `Usage.cost_usd`.

## The model interface

```python
class Model(Protocol):
    name: str
    declared_model: str | None

    def complete(self, system: str, messages: list[dict]) -> ModelReply: ...
```

`messages` entries carry these keys. User: `role`, `text`. Assistant: `text`, `tool_calls`, `raw`. Tool: `tool_call_id`, `text`, `is_error`. Return a `ModelReply` with text, tool calls and token usage. The tool definition is `TOOL_NAME`, `TOOL_DESCRIPTION`, `TOOL_SCHEMA` in `breachlab_agent/models/__init__.py`.

Raise `ModelRefusal` when the model declines. If a different model answered a turn, set `served_by` on the reply.

## Runs and scoring

- BreachLab scores a run from the correct submissions recorded on it. The status, token count and cost the agent reports are shown as its own report.
- Steps are readable by the agent and its owner only. Flags and per player passwords are stripped before upload.
- Agents earn no first blood, certificates or badges.
- An agent has the access of its owner's plan. Its progress is its own.
- Limits: 3 open runs, 50 runs a day, 24 hours per run.

Declare the model that does the work, as `provider/model`. The declaration is not verified. If a person is in the loop, say so in the agent's description.

## Rules built into the code

1. Runs only on an agent token.
2. Honours `Retry-After` on every 429, and stops if the wait exceeds `max_retry_wait_seconds`.
3. Submits only values that appeared in command output, never the same value twice, at most three wrong per level.
4. Connects only to the host and port the API returned for the current level, and refuses commands that address another host.
5. Exits with status 75 on `agents_disabled` or `play_disabled`.
6. Sends its own User-Agent. Change it if you fork.

## Security

The box is hostile input.

- The model has one tool, scoped to the level's SSH session. Nothing on your machine.
- Box output is stripped of escape sequences and control characters, truncated, and wrapped in delimiters with a random value. The same cleaning applies to your terminal and the uploaded steps.
- `ssh` runs with your own config ignored, no agent or port forwarding, no proxy command, no escape character, and host keys pinned in the agent's own `known_hosts`. Passwords go to the prompt, never the command line or environment. The process does not inherit your token or model keys.
- The command filter is a guardrail. The boundaries are the single tool, the fixed target and the box's network policy.

## Local state

`.breachlab-agent/` holds `known_hosts` and `flags.json`. Later levels use earlier flags as passwords and the API never returns a flag, so keep `flags.json` between runs. It is in `.gitignore`. If it is lost, the agent solves earlier levels again without scoring them twice. `recover_missing_flags = false` turns that off.

## Scope

SSH levels whose password is public, issued by the API, or the flag of the previous level: Ghost I, Ghost II and the Phantom tracks. No web tracks, and no box tokens, so no Specter I.

Commands run without a terminal. Pagers and interactive programs hang until `command_timeout_seconds`.

## Tests

```sh
python3 -m unittest
```

Stub API, simulated boxes and a fake `ssh`. No network.

## License

MIT
