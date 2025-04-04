# aquordion
A quantum library to test and benchmark Pasqal's backends.

## Cross-backend testing

Run the following to check correctness between backends:

```bash
hatch -e tests run test
```

## Timing benchmarks

Run the following command to generate timings:

```bash
hatch -e tests run benchmarks
```
