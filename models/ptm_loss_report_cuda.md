## Supported Modifications

Values are `%RMSE vs model average RMSE` on each model holdout/test set.
`✅` indicates `<120%` and `>1000` PTM occurrences in that model's training split; otherwise `⚠️`.

| Modification | Sites | UNIMOD | Chronologer | Cartographer | Electrician | Sculptor |
| --- | --- | --- | --- | --- | --- | --- |
| Unmodified | - | - | ✅ 79.1% | ✅ 100.4% | ✅ 112.1% | ✅ 99.4% |
| Acetyl | K, N-term | 1 | ⚠️ 181.2% | ✅ 100.1% | ✅ 92.7% | ✅ 106.6% |
| Carbamidomethyl | C | 4 | ✅ 108.7% | ✅ 88.4% | ✅ 99.3% | ✅ 118.3% |
| Deamidation | N, Q, R | 7 | N/A | ✅ 89.8% | ✅ 111.7% | ⚠️ 123.4% |
| Dimethyl | K, R | 36 | ⚠️ 308.9% | N/A | N/A | ⚠️ 88.8% |
| GlyGly (Ub) | K | 121 | ✅ 58.0% | ✅ 86.7% | ⚠️ 122.2% | ✅ 93.6% |
| HexNAc | N, S, T | 43 | N/A | ✅ 88.9% | ⚠️ 144.5% | ⚠️ 106.4% |
| Methyl | K, R | 34 | ⚠️ 221.7% | ✅ 91.2% | ✅ 97.1% | ⚠️ 92.2% |
| Oxidation | M, W | 35 | ✅ 118.8% | ✅ 97.8% | ✅ 101.5% | ✅ 107.9% |
| Phospho | S, T, Y | 21 | ⚠️ 249.7% | ✅ 96.9% | ✅ 92.1% | ✅ 94.7% |
| Succinyl | K | 64 | ⚠️ 42.5% | N/A | N/A | ⚠️ 75.3% |
| Trimethyl | K | 37 | ⚠️ 445.1% | N/A | N/A | ⚠️ 64.0% |
| Pyro-Glu | Q (N-term), E (N-term) | 28, 27 | ✅ 115.2% | ✅ 113.3% | ✅ 58.4% | N/A |
| TMT0 | K, N-term | 739 | ⚠️ 148.2% | N/A | N/A | N/A |
| TMT6plex | K, N-term | 737 | ⚠️ 150.7% | ✅ 97.6% | ✅ 83.9% | N/A |
| Biotin | K | 3 | N/A | N/A | N/A | ⚠️ 41.9% |
| Butyryl | K | 1289 | N/A | N/A | N/A | ⚠️ 94.4% |
| Crotonyl | K | 1363 | N/A | N/A | N/A | ⚠️ 75.1% |
| Cysteinyl | C | 312 | N/A | N/A | N/A | ⚠️ 65.6% |
| Formyl | K | 122 | N/A | N/A | N/A | ⚠️ 79.2% |
| Glutarylation | K | 1848 | N/A | N/A | N/A | ⚠️ 86.2% |
| Glycosyl hydroxyproline | P | 408 | N/A | N/A | N/A | ⚠️ 115.1% |
| Hydroxyisobutyryl | K | 1849 | N/A | N/A | N/A | ⚠️ 89.0% |
| Malonyl | K | 747 | N/A | N/A | N/A | ✅ 67.3% |
| Nitro | Y | 354 | N/A | N/A | N/A | ⚠️ 49.0% |
| Propionyl | K, N-term | 58 | N/A | N/A | N/A | ⚠️ 78.3% |

### Average Model Loss (RMSE Baseline)

- `Chronologer` average `RMSE(HI)` = `1.281231`
- `Cartographer` average `RMSE(fragment_intensity)` = `0.142406`
- `Electrician` average `RMSE(charge_state_dist)` = `0.097929`
- `Sculptor` average `RMSE(CCS)` = `15.720550`
