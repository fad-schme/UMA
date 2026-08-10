# uma/retrieve/rlm/__init__.py
#
# Deliberately empty. Import RLM pieces from their modules
# (`uma.retrieve.rlm.controller`, `.request`, `.context_pack`, ...) — every
# caller in the tree already does.
#
# Re-exporting them here pulled the whole controller chain in whenever a leaf
# module was imported, which closed two import cycles: `planner` ->
# `rlm.intent` -> (this file) -> `controller` -> `evidence` -> `request` ->
# `planner`, and `uma.memory.chunk.core` -> `uma.retrieve.ranking` -> (this
# file) -> `controller` -> `uma.memory.chunk.core`. Both made a plain
# `import uma.memory` fail in a fresh interpreter. Keep this file import-free.
