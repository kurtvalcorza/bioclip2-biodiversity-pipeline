# Offline stage-chain check, revision 0.2.2-candidate

`prepare_synthetic_harness.py <materialised carrier> <run root>` copies the carried files, replaces every
photograph with a random image of the manifest's dimensions (rewriting bytes, SHA-256 and dHash and
dropping the manifest digest), pre-fills the cache and runs `capstone.py --stage prepare`.
`stage_chain_harness.py --root <run root> --stage <stage>` then runs each later stage in its own
process with deterministic stub encoders instead of BioCLIP and SigLIP. `results.json` summarises the
run. Synthetic inputs and stub models: this is software evidence only.
