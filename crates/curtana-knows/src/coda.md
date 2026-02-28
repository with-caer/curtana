# `Curtana` Coda

Data structures for knowledge graph storage and retrieval.

## `Artifact` Data

+ `id` text

    The unique identifier of the artifact in the system.

+ `timestamp` u64

    The time the artifact was created. If creation time is unknown, the time it was ingested into storage is used.

+ `author` text

    The name of the person or entity that authored the artifact.

+ `contents` unspecified

    The raw contents of the artifact.

+ `embedding` optional list of f32

    A vector embedding over the contents of the artifact.
