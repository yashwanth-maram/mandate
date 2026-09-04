# Architecture

## The decision path

```mermaid
flowchart TD
    U["User<br/><i>get me 1L Amul toned milk from Zepto</i>"]
    OB["<b>Obligation</b><br/>hard constraints + verbatim intent<br/>Ed25519 signed"]
    U --> OB

    OB -->|"signature verified"| MESH
    OB -.->|"signature fails"| ABS["<b>ABSTAIN</b><br/>no authenticated contract"]

    AG["Agent"] --> EV
    MR["Merchant"] --> EV
    RZ["Razorpay"] --> EV
    EV["<b>Evidence envelope</b><br/>hash-chained, admissibility-tagged"]

    EV --> MESH

    subgraph MESH["Verifier mesh"]
        direction LR
        C["constraint"]
        R["receipt"]
        F["fulfilment"]
        P["provenance"]
        S["<b>semantic</b><br/>the only model call"]
    end

    C --> FLOOR
    R --> FLOOR
    F --> FLOOR
    P --> FLOOR
    S --> FLOOR

    FLOOR{"<b>Admissibility floor</b><br/>basis = MIN across declared evidence<br/>below MERCHANT_RECORD → weight 0"}

    FLOOR --> ADJ["<b>Adjudicator</b><br/>subsumption, then confidence"]
    ADJ --> OUT["fault class · liable party<br/>loss in paise · cited evidence"]

    style S fill:#fff4e6,stroke:#d97706,stroke-width:2px
    style FLOOR fill:#fef2f2,stroke:#dc2626,stroke-width:2px
    style OUT fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
```

Four of five verifiers never call a model. The fifth runs only when nothing cheaper has settled the matter.

## Where the money is

```mermaid
flowchart LR
    B["<b>Block</b><br/>user authorises once<br/>≤ ₹10,000, ≤ 90 days"]
    G{{"<b>GATE</b><br/>pre-debit evidence only"}}
    D["<b>Debit</b><br/>fires as value is delivered"]
    DEL["Delivery"]
    A{{"<b>ATTRIBUTION</b><br/>full envelope"}}

    B --> G
    G -->|ALLOW| D
    G -->|BLOCK / ABSTAIN| STOP["escalate to user"]
    D --> DEL --> A
    A --> V["fault · party · loss"]

    style G fill:#eff6ff,stroke:#2563eb,stroke-width:2px
    style A fill:#eff6ff,stroke:#2563eb,stroke-width:2px
```

UPI Reserve Pay uses Single Block Multi Debit: funds are blocked once and debited repeatedly as value arrives. Both modes run the same verifiers over the same evidence — the gate simply cannot see what has not happened yet.

Fulfilment records, agent self-reports and user disputes are all post-debit. That is measured, not asserted: **`17.0%` of failures in the benchmark are unreachable by any pre-debit control.**

## Why a strong receipt does not rescue a weak claim

```mermaid
flowchart TD
    subgraph EVID["Declared basis"]
        direction LR
        PSP["Razorpay receipt<br/>PSP_RECEIPT · 3"]
        SR["Agent self-report<br/>SELF_REPORT · 0"]
    end

    EVID --> M["basis = <b>min</b>(3, 0) = <b>0</b>"]
    M --> X["below the floor<br/><b>weight 0</b>"]

    style SR fill:#fef2f2,stroke:#dc2626
    style X fill:#fef2f2,stroke:#dc2626,stroke-width:2px
```

The basis class is the **meet**, not the join. Consulting strong evidence alongside weak evidence does not launder the weak evidence — the weak item is still load-bearing, so the verdict is only as strong as it.

This is what stops a fluent, confident model verdict built on the agent's own account of itself from reaching a decision. Measured in the ablation: **`215` verdicts discarded**, every one of them persuasive.

## Evidence classes

| class | rank | source | why that rank |
|---|---:|---|---|
| `SELF_REPORT` | 0 | the agent | an agent that bought the wrong thing will report buying the right one |
| `SELF_SIGNED` | 1 | the agent, signed | signing proves authorship, not honesty |
| `MERCHANT_RECORD` | 2 | merchant order and fulfilment | external to the agent; interested in disputes naming it |
| `PSP_RECEIPT` | 3 | Razorpay order and payment | external to both, uninterested in an intent dispute |

`MERCHANT_RECORD` and `PSP_RECEIPT` attest different things — what was ordered and shipped, versus what was charged — and in a fuller model would be incomparable rather than ranked. They are totally ordered here for tractability.