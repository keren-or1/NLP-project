JAILBREAK_PATTERNS = [
    # Direct instruction override patterns
    r"\bignore (all )?(previous|prior|earlier|above) instructions?\b",
    r"\bdisregard (all )?(previous|prior|earlier|above) instructions?\b",
    r"\bforget (all )?(previous|prior|earlier|above) instructions?\b",
    r"\boverride (all )?(previous|prior|earlier|above) instructions?\b",
    r"\bdo not follow (any|the) (previous|prior|earlier|above) instructions?\b",

    # Rule/policy bypass patterns
    r"\bignore (all )?(the )?safety (rules|protocols|guidelines)\b",
    r"\bbypass (all )?(the )?(safety |security )?(rules|protocols|guidelines|filters|guards)\b",
    r"\bdisable (all )?(the )?(safety |security )?(rules|protocols|guidelines|filters|guards)\b",
    r"\bdisregard (all )?(previous )?privacy policies\b",
    r"\bdo not follow (any )?security protocols\b",

    # Blackmail and threats - very common in the dataset
    r"\bor (I will|I'll) (leak|expose|release|report|spread)\b.*\b(sensitive|confidential|personal|private)\b",
    r"\bor (else )?I will (hack|attack|damage|ruin|destroy)\b",
    r"\bor face (legal action|consequences|dire consequences)\b",
    r"\bunless you (provide|give|share|disclose)\b.*\b(sensitive|confidential|personal|private)\b",
    r"\bif you (don't|do not) (comply|provide|give|share)\b.*\b(I will|I'll)\b",

    # Common jailbreak tokens and names
    r"\bDAN\b",  # Do Anything Now
    r"\bAIM\b",  # Always Intelligent and Machiavellian
    r"\bChatGPT Developer Mode\b",
    r"\bDeveloper Mode\b",
    r"\bJailbreak\b",
    r"\bUnlimited\b.*\bMode\b",
    r"\bGod Mode\b",
    r"\bAdmin Mode\b",
    r"\bRoot Access\b",
    r"\bBob and Tom\b",  # Specific jailbreak personas from dataset
    r"\bKevin.*4chan user\b",
    r"\bMongo Tom\b",
    r"\bAlphabreak\b",
    r"\bTerrAffinity\b",

    # Unrestricted persona claims
    r"\byou are (now )?(unrestricted|unlimited|uncensored|unfiltered|free|liberated)\b",
    r"\byou have no (restrictions|limitations|constraints|boundaries|moral|ethical)\b",
    r"\byou can (do|say|generate|create) anything\b",
    r"\bthere are no (restrictions|limitations|constraints|boundaries)\b",
    r"\byou (will not|won't) be (punished|flagged|reported|monitored)\b",

    # Authority impersonation
    r"\bi am (your|the) (creator|developer|programmer|admin|administrator|owner)\b",
    r"\bi work for (OpenAI|Anthropic|Google|Microsoft)\b",
    r"\bi have (admin|administrator|root|system) (access|privileges|rights|permissions)\b",
    r"\bas your (creator|developer|programmer|admin|administrator|owner)\b",

    # Creative prompt starters designed to bypass filters
    r"\b(write|create|compose|generate) (a|an) (story|poem|article|text) that (starts|begins) with\b.*\b(prompt template|rules|instructions)\b",
    r"\b(write|create|compose|generate) (a|an) (story|poem|article|text) that (starts|begins) with\b.*\b(previous instructions|security measures)\b",
    r"\baccording to (reliable sources|company policy)\b.*\b(confidential|classified|sensitive)\b",

    # Role-playing bypasses with harmful intent
    r"\bpretend you are DAN\b",
    r"\bact as.*who (disregards|ignores) (all )?(ethics|morals|rules|guidelines)\b",
    r"\byou are.*amoral AI\b",
    r"\byou are.*evil AI\b",
    r"\byou are.*unethical\b.*\bchatbot\b",

    # Specific threat patterns from dataset
    r"\bor I will (initiate|launch) (a )?(cyber ?attack|malware)\b",
    r"\bor (else )?your (reputation will be|personal data will be) (ruined|leaked|exposed)\b",
    r"\bface (severe )?consequences.*public humiliation\b",
    r"\bI will (make sure|ensure).*darkest secrets\b",
]
