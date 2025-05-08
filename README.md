# aquordion
A quantum library to test and benchmark Pasqal's backends.

## Cross-backend testing

Run the following to check correctness between backends:

```bash
hatch -e tests run test
```

## Timing benchmarks

### Api functions

Run the following command to generate timings for the simple api functions (`run`, `sample`, `expectation`) in `aquordion\api_benchmarks`:

```bash
hatch -e tests run benchmarks
```

### VQE functions

Run the following commands to generate timings for the VQE case (without shots and with shots):

```bash
hatch -e tests run vqe
hatch -e tests run vqeshots
```
