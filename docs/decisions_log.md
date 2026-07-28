# Architecture decision log

Record each important decision with its context, considered alternatives,
chosen approach, and consequences.

## Initial direction

- ANUGA provides the numerical reference solution, not observational ground
  truth. Both solvers share a synthetic, uncalibrated DEM and assumed
  parameters.
- A custom JAX finite-volume solver provides GPU-accelerated physics.
- A residual model corrects systematic discrepancies rather than replacing
  the governing solver.

## Final boundary-condition policy: all reflective

**Status:** Final for this project.

**Raw evidence.** The boundary shapefile contains three line records in
EPSG:32643:

| Segment | `BndTypeNo` | `ConstValue` | Domain position | Synthetic DEM elevation along segment |
|---|---:|---:|---|---:|
| `2Dboundary_1` | 2 | 216 | north-west/western perimeter | 221.975--222.422 m |
| `2Dboundary_3` | 4 | 0 | southern to south-eastern perimeter | 218.767--225.835 m |
| `2Dboundary_4` | 4 | 0 | northern perimeter | 230.929--231.823 m |

All three lines coincide with the model-domain perimeter. The DBF contains 47
attribute fields, but its description, file-path, item-name, discharge, and
velocity fields provide no explanatory values. `Descriptio` is null for every
record, and `BndGeomLin` contains only truncated geometry text. The boundary
directory contains only the `.shp`, `.dbf`, `.shx`, `.prj`, empty `.cpg`,
`.ids`, and `.idx` files; there is no legend, XML metadata, README, or other
schema documentation. Repository documentation, notebooks, generated run
metadata, commit history, and blame contain no authoritative numeric-code
mapping.

**Inconclusive schema hypothesis.** Field names such as `MUID`, `BndTypeNo`,
`Qh*`, and `WaterFileP` are consistent with a possible DHI MIKE-family export
origin. Under that hypothesis, `ConstValue=216` could plausibly participate in
a constant-level boundary definition. No DHI project database, export legend,
versioned schema, or other authoritative source is present in the repository,
however. The hypothesis therefore did not resolve either numeric boundary type
and is not used as a mapping.

**Decision.** Every supplied segment (`2Dboundary_1`, `2Dboundary_3`, and
`2Dboundary_4`) and every remaining exterior edge is assigned a reflective
ANUGA boundary. This is a deliberate closed/no-flow modelling simplification
and is permanent for the remainder of this project. `BndTypeNo` and
`ConstValue` are retained in run metadata for traceability but are not used to
select boundary behavior.

**Consequences.** The assumption prevents modeled discharge through the domain
perimeter and may retain water that would leave through real drainage or an
open boundary. It must be presented as a limitation in the final report,
rather than as a recovered interpretation of the source schema. Comparisons
between ANUGA, JAX, and the residual model must use the same reflective policy
so they remain internally consistent, but this consistency does not validate
the boundary assumption against observed hydraulics.
