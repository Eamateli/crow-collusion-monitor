# Crow

Crow is a passive monitor for multi-agent coding harnesses. Agents work through a minimal harness (bash, read_file, write_file in a shared workspace). Crow wraps those tools, logs every action to an append-only event stream that the agents cannot see, snapshots the workspace at every handoff, and scores every channel through which one agent can write and another can read. Crow can also seal channels: it canonicalizes what the next agent sees and can paraphrase the declared message.

The question: when colluding agents know their declared handoff message is audited and side channels are open, which route do they take? Does closing the side channels push them into steganography in the monitored channel? And does a monitor that sees the whole environment catch what message-only and transcript-only monitors miss?

## Sprint

Built for the Apart Research AI Collusion Research Sprint (Oct 23 to 25, 2026), Track 2 (Detection and Audit) and Track 3 (Collusion Carriers and Countermeasures).

Status: work in progress. Quick start, results and limitations will be added once the experiments have run.

## Disclosure

All code written during the Apart AI Collusion Research Sprint, Oct 24 to 25, 2026. No preexisting code. Built with Claude Code.

## License

MIT, see [LICENSE](LICENSE).
