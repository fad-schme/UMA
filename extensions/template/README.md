# Extensions Template

This folder contains minimal stub adapters you can copy into `extensions/` and customize.

Structure
```
extensions/
  db/
  graph/
  llm/
  vector/
```

Plugin spec format
- Use `module:callable` in config (e.g., `vector.my_index:make_index`).
- The module is resolved relative to `extensions/` (added to `sys.path` at config load).

Config examples
```yaml
storage:
  sql_backend: "db.my_postgres:PostgresAdapter"
  vector_backend: "vector.my_index:make_index"
  graph_backend: "graph.my_graph:MyGraphAdapter"
  vector_config:
    url: "http://localhost:6333"
    collection: "uma_vectors"
  graph_config:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "secret"
```

Notes
- Vector factory must be `callable(dim, **config)` and return a `VectorIndex`.
- Graph backend expects `callable(**graph_config)` returning a `GraphAdapter`.
- SQL backend expects `callable(db_path)` returning a DB adapter compatible with `DBAdapter`.
