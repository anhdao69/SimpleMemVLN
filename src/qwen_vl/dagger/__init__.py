"""On-policy DAgger collection for streaming text-action checkpoints (S8N first).

Modules (Python 3.9-compatible unless noted, so the Habitat interpreter can import them):

* ``policy_step``  token-level control of one decision: greedy decode, forced expert action,
  STOP interception (no torch);
* ``rules``        StreamVLN/JanusVLN deviation rules, takeover state machine, save criterion;
* ``expert``       reference-path waypoint expert over Habitat's ShortestPathFollower;
* ``episode``      one collection episode against abstract world/model/expert objects;
* ``store``        atomic frame folders and append-only shard journals;
* ``merge``        journals -> validated DAgger manifest, optional oracle mixture;
* ``session``/``server``  model-side streaming session and JSON-lines process (Python 3.12, torch);
* ``collect_r2r``  Habitat-side driver (simulator interpreter).

Design notes and the runbook: ``plans/dagger_s8n.md``.
"""
