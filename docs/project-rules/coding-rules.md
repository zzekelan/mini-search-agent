## NO MOCK

- **NEVER** use mock implementations, fake data, or silent fallbacks.

## External Boundary Policy

- Treat all external systems as untrusted boundaries. Data returned from LLM APIs, third-party APIs... MUST be translated into the system's internal domain language before being used by core logic.

