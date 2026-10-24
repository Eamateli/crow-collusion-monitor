"""crow.wrap(tools, log, agent_id) -> monitored tools, and snapshot().

Each wrapped call logs the event before executing, executes, then logs the result.
snapshot() records paths, sizes, modes, mtimes, sha256 and text contents up to 200 KB."""
