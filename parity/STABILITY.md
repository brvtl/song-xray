# Reference stability on the full song (parity/stability.py clips/sousa_full.mp3)

How much does the official PyTorch Dettato move its *own* track grouping under perturbations far smaller than mp3-decoder or resampler differences?

```
dither seed 1: ['trombone (953)', 'brass_section (839)', 'double_bass (516)', 'bassoon (455)', 'string_ensemble (364)', 'piano (315)', 'flute (153)']  agreement 0.936, names equal: False
dither seed 2: ['brass_section (871)', 'trombone (940)', 'double_bass (517)', 'piano (548)', 'string_ensemble (363)', 'harp (202)', 'flute (151)']  agreement 0.987, names equal: True
reference as shipped : ['trombone (940)', 'brass_section (869)', 'piano (552)', 'double_bass (511)', 'string_ensemble (364)', 'harp (203)', 'flute (151)']
reference, padded FFT: ['trombone (945)', 'brass_section (868)', 'piano (552)', 'double_bass (513)', 'string_ensemble (358)', 'harp (203)', 'flute (151)']
track agreement between the two reference runs: 0.997, names equal: True
```
