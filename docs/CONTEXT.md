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

**Parallel Tool Execution**:
A **Turn** behavior where multiple independent tool calls emitted by one model response may execute concurrently while their results remain ordered in the **Session Timeline** by the original tool-call order.
_Avoid_: speculative execution, background tools, cross-turn concurrency

**Parallel-Safe Tool Call**:
A tool call that may run at the same time as other **Parallel-Safe Tool Calls** emitted by the same model response without changing the research meaning of the **Turn**.
_Avoid_: any fast tool, any read-looking command

**Session**:
A long-lived Mini Search Agent workspace that contains durable source artifacts and replayable agent work for one research task.
_Avoid_: Turn, one command invocation

**Session Timeline**:
The durable, replayable content history of a **Session**. It records what the agent saw, said, requested from tools, and received from tools without being a per-turn execution log.
_Avoid_: Turn history, execution log, run folder

**Timeline Entry**:
An ordered item in a **Session Timeline** that may be produced by a **Turn**.
_Avoid_: naked Message, Turn message, Timeline Item

**Timeline Part**:
A typed content block inside a **Timeline Entry**, such as text, tool call, or tool result.
_Avoid_: top-level timeline item, standalone history item, provider role as domain role

**Tool Result Error**:
A tool-result **Timeline Part** that records a failed, unavailable, malformed, or otherwise unsuccessful tool call so the model can recover later in the same **Turn**.
_Avoid_: Turn failure, bare exception, missing tool result

**Turn**:
An execution lifecycle for one user prompt while answering or continuing a **Research Question**. A Turn includes the LLM calls, tool calls, and final answer or failure produced directly by that agent; a subagent appears in its parent Turn as a subagent tool call and has its own **Spawned Turn** in a **Sub-session**.
_Avoid_: Run, timeline owner, conversation message

**Turn Console View**:
A temporary command-line projection of a **Turn** for a human operator. It may show **Main Agent** streaming text and per-tool-call progress from stable **Turn** events, but it is not part of the **Session Timeline**.
_Avoid_: Session log, Timeline UI, transcript

**Sub-session**:
A child **Session** created to isolate Search Subagent work from its parent **Session**.
_Avoid_: Fork, branch

**Spawned Turn**:
A **Turn** in a **Sub-session** that was started by a parent **Turn** through the `subagent` tool.
_Avoid_: child Turn owned by parent Turn

**Candidate URL**:
A URL discovered by search before direct fetch and content verification. A Candidate URL is not evidence until it becomes a Source Note.
_Avoid_: Source, citation

**Source Note**:
A deduped, fetched, and verified source summary with URL, fetch status, reliability marking, search queries, and evidence. Source Notes are the evidence base for Final Claims.
_Avoid_: Search result, source summary

**Final Claim**:
A statement in the final answer that cites one or more Source Notes. Final Claims are the auditable bridge between the answer and the fetched evidence.
_Avoid_: Bullet point, takeaway

**Eval Trace**:
A durable execution record for evaluating one **Session**. It follows the **Session** and **Sub-session** hierarchy, is composed of nested **Eval Spans**, distinguishes **Main Eval Trace** from **Sub Eval Trace**, and is not the model-visible **Session Timeline** or the compact **Eval Report**.
_Avoid_: Timeline, telemetry, Eval Report, transcript

**Main Eval Trace**:
The **Eval Trace** for the **Main Agent** work in the main **Session**.
_Avoid_: parent Turn log, Session Timeline

**Sub Eval Trace**:
The **Eval Trace** for **Search Subagent** work in a **Sub-session**. It sits in the parent **Session** hierarchy and is linked from the parent **Main Eval Trace** through the subagent tool-call span.
_Avoid_: fork trace, inline parent span

**Eval Span**:
An evaluable unit inside an **Eval Trace**. Eval Spans are nested; the three canonical types are **Turn Span**, **LLM Call Span**, and **Tool Call Span**.
_Avoid_: Timeline Entry, Telemetry event, flat step

**Turn Span**:
The top-level **Eval Span** covering one **Turn**: one user prompt and all LLM/tool-call loops needed to produce that Turn's final answer or failure.
_Avoid_: Run Span, Timeline Entry, assistant message

**LLM Call Span**:
An **Eval Span** for one model call inside a **Turn Span**. It may contain zero or more child **Tool Call Spans** from that call's response.
_Avoid_: Provider request, assistant Timeline Entry alone

**Tool Call Span**:
An **Eval Span** for one tool execution. It is always a child of an **LLM Call Span**. For a subagent tool call it may contain further child **LLM Call Spans** and **Tool Call Spans**.
_Avoid_: Standalone tool call, timeline part alone

**Code Eval**:
An eval that uses deterministic code to check structural or format constraints. It produces a binary score and a label.
_Avoid_: Lint, unit test

**LLM Judge**:
An eval that uses an LLM to judge semantic quality against a rubric. It produces a score, a label, and an explanation.
_Avoid_: Human review, deterministic check

**Eval Report**:
A compact self-check describing query coverage, fetch success, source reliability mix, citation coverage, and known limitations of a turn.
_Avoid_: Score, benchmark

**Research Artifact**:
A durable source record written during a turn, such as the source index or individual Source Notes. Research Artifacts are the auditable source trail, not temporary agent messages or final-answer files.
_Avoid_: Scratch note, transcript

## Example Dialogue

Developer: "The Main Agent made four query angles, but two Search Subagents returned the same article."

Domain expert: "Keep one Source Note for that article and record both query angles on it. The Final Claim should cite the Source Note, not the duplicate Candidate URLs."

Developer: "A search result looked relevant but the page failed to fetch."

Domain expert: "Then it remains a Candidate URL with failed fetch status. It cannot support a Final Claim."

Developer: "The Search Subagent mentioned a useful page in its message but no Research Artifact captured it."

Domain expert: "Then the source trail is not auditable yet. The Main Agent should record the Source Note and cite that Research Artifact, not the transient message."
