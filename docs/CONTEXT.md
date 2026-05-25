# Mini Search Agent

A research-assistant context for answering open research questions from fetched, verified, and cited web sources.

## Language

**Research Question**:
An open-ended question submitted by the user for investigation. It is broader than a keyword search and should require synthesis across multiple sources.
_Avoid_: Prompt, search term

**Main Agent**:
The coordinating agent responsible for turning a Research Question into a Query Plan, dispatching Search Subagents, deduping Source Notes, and producing cited Final Claims.
_Avoid_: Orchestrator, planner bot

**Query Plan**:
A set of search angles derived from the Research Question. Each angle has a purpose and a concrete query string.
_Avoid_: Search list, task list

**Search Subagent**:
A worker assigned to one query angle. It searches, fetches candidate URLs, verifies content, and returns Source Notes to the Main Agent.
_Avoid_: Search worker, crawler

**Tool Filter**:
A runtime capability boundary that gives each agent role only the tools it is allowed to use. It limits available actions but does not judge whether the research output is complete or correct.
_Avoid_: Eval guardrail, quality checker

**Session**:
A long-lived Mini Search Agent workspace that contains durable source artifacts and replayable agent work for one research task.
_Avoid_: Run, one command invocation

**Session Timeline**:
The durable, replayable content history of a **Session**. It records what the agent saw, said, requested from tools, and received from tools without being a per-run execution log.
_Avoid_: Run history, execution log, run folder

**Timeline Entry**:
An ordered item in a **Session Timeline** that may be produced by a **Run**.
_Avoid_: naked Message, Run message, Timeline Item

**Timeline Part**:
A typed content block inside a **Timeline Entry**, such as text, tool call, or tool result.
_Avoid_: top-level timeline item, standalone history item, provider role as domain role

**Run**:
An execution lifecycle that may produce **Timeline Entries** while answering or continuing a **Research Question**.
_Avoid_: Turn, timeline owner, conversation message

**Sub-session**:
A child **Session** created to isolate Search Subagent work from its parent **Session**.
_Avoid_: Fork, branch

**Spawned Run**:
A **Run** in a **Sub-session** that was started by a parent **Run** through the `subagent` tool.
_Avoid_: child Run owned by parent Run

**Candidate URL**:
A URL discovered by search before direct fetch and content verification. A Candidate URL is not evidence until it becomes a Source Note.
_Avoid_: Source, citation

**Source Note**:
A deduped, fetched, and verified source summary with URL, fetch status, reliability marking, search queries, and evidence. Source Notes are the evidence base for Final Claims.
_Avoid_: Search result, source summary

**Final Claim**:
A statement in the final answer that cites one or more Source Notes. Final Claims are the auditable bridge between the answer and the fetched evidence.
_Avoid_: Bullet point, takeaway

**Eval Report**:
A compact self-check describing query coverage, fetch success, source reliability mix, citation coverage, and known limitations of a run.
_Avoid_: Score, benchmark

**Research Artifact**:
A durable source record written during a run, such as the source index or individual Source Notes. Research Artifacts are the auditable source trail, not temporary agent messages or final-answer files.
_Avoid_: Scratch note, transcript

## Example Dialogue

Developer: "The Main Agent made four query angles, but two Search Subagents returned the same article."

Domain expert: "Keep one Source Note for that article and record both query angles on it. The Final Claim should cite the Source Note, not the duplicate Candidate URLs."

Developer: "A search result looked relevant but the page failed to fetch."

Domain expert: "Then it remains a Candidate URL with failed fetch status. It cannot support a Final Claim."

Developer: "The Search Subagent mentioned a useful page in its message but no Research Artifact captured it."

Domain expert: "Then the source trail is not auditable yet. The Main Agent should record the Source Note and cite that Research Artifact, not the transient message."
