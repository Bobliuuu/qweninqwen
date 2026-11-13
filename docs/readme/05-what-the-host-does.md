| Role | Why |
|---|---|
| Boot, the `GO` token, the embed-and-append loop, `ecall` I/O | 🟦 allowed by the design |
| Comparison against the reference trace | 🟦 the test, not an input |
| Later chunks' start state | 🟨 seeded from the reference, verified by exact end-state equality at all 95 seams |

Everything else runs in the circuit. Full audit with file and line references: [`docs/HOST_ROLE_AUDIT_CFRAM.md`](docs/HOST_ROLE_AUDIT_CFRAM.md).
