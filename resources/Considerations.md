## Time per move availability

100 million move checks, Time $ 50h  $, 256 processes

Time per move
$$
 \frac{50 \times 3600s \times 256}{ 1e7} \approx 0.5s
$$
Acceptable!

## Data Send limitations

Input size:
$$
 8 \times 4 \times 52 = 1664
$$
Output size:
$$
 8 \times 4 \times 33 = 1056
$$
assuming float32:
$$
 (1664 + 1056 ) \times 4 \mathrm{B} = 10.88 \mathrm{KB}
$$

For 1 Million games $\approx 1.1$ TB for 100Gb/s about $ 80 $ s under ideal conditions.

Input size:
$$
 8 \times 4 \times 52 = 1664
$$
Output size:
$$
 8 \times 4 \times 33 = 1056
$$
assuming float32:
$$
 (1664 + 1056 ) \times 4 \mathrm{B} = 10.88 \mathrm{KB}
$$

For 1 Million states $\approx 1.1$ TB for 100Gb/s about $80$s under ideal conditions.
Expect 100 states maybe: 8000s -> 2h of only data transfer at full run

-> **Evaluation to be done on node, Training does not matter as much.**
