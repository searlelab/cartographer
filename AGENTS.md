# AGENTS.md
Agent guidance for this repository.

## Read-first
- Before making changes, read `README.md` at least once to understand architecture, module boundaries, and existing patterns.

## Repo layout (high level)
- Maven multi-module project: parent (root `pom.xml`) with submodules:
  - `core/` (Java core library)
  - `gui/` (Swing GUI tools)
- Native/vendor integrations:
  - `rust-jni/`: Rust JNI bridge for Bruker readers, exposed to Java via JNI
  - `thermo-raw-server/`: C# .NET gRPC server for Thermo RAW
  - `proto/`: protobuf definitions and build artifacts used for Java <-> C# interface

## Build and test commands
### Fast compile, no native rebuild, no tests
mvn -pl core,gui -am -Dskip.build.natives=true -Dmaven.test.skip=true compile

### Full test suite (core + gui), skip native rebuild
mvn -pl core,gui -am -Dskip.build.natives=true test

### Recommended: run selected tests
- Focus a module:
  - `mvn -pl core -am -Dskip.build.natives=true -Dtest=<TestClass> test`
  - `mvn -pl gui  -am -Dskip.build.natives=true -Dtest=<TestClass> test`
- Focus a single test method (Surefire):
  - `mvn -pl core -am -Dskip.build.natives=true -Dtest=<TestClass>#<methodName> test`

## What “good autonomy” looks like
- Prefer changes in `core/` and `gui/` unless the task explicitly requires native or server changes.
- Reuse existing code paths and utilities, especially for file parsing, object models, windowing, and output writers.
- Do not speculate about code you have not opened. If a conclusion depends on specific behavior, open and read the relevant files first (source, tests, build config, scripts).
-  Minimize blast radius:
  - Change the fewest files possible.
  - Keep diffs small and easy to review.
  - Prefer additive changes over rewiring existing behavior.
- Make success checkable:
  - Add or update a runnable test, or a deterministic golden-output check using fixtures under `core/src/test/resources` or `gui/src/test/resources`.
  - Ensure outputs are deterministic (ordering, rounding/epsilon, fixed seeds if randomness exists).

## Coding standards
- Preserve formatting and conventions already present in nearby files.
- Preserve and maintain comments, update comments when behavior changes and use the same comment style as nearby code.

## Swing rules (gui/)
- UI updates must happen on the Swing EDT.
- Long-running work must not block the EDT, use existing background patterns already present in the codebase.
- Prefer testing state, view-model logic, and data model transformations in `gui/` without timing sleeps.

## Native and cross-language boundaries (only if needed)
- If a task requires Bruker changes, scope work to `rust-jni/` plus the Java JNI interface layer, keep the Java public surface stable.
- If a task requires Thermo changes, scope work to `thermo-raw-server/` and `proto/`, keep the Java client contract stable.
- Do not introduce new build steps, toolchains, or dependencies unless explicitly requested.

## Final output expectation
Always attempt to compile the Java code before finishing. When done, report:
- Commands you ran (including focused tests and/or full tests)
- What changed (files/modules)
- How correctness was verified (test names, fixtures, golden checks)
- When you make a claim about behavior, include the evidence path: file names and the exact methods or sections you relied on.
