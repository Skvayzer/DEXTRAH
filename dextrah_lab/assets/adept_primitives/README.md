# ADEPT primitive assets

The 16 simulation USD files in `USD/` are generated from the dimensions shown
in ADEPT v1 Appendix Fig. 8. They are not checked in because USD serialization
depends on the installed OpenUSD version.

Generate them inside the Isaac Sim environment with:

```bash
python scripts/generate_adept_primitives.py
```

The corresponding deterministic 64-point surface representations are defined
in `dextrah_lab/tasks/dextrah_kuka_allegro/adept_mdp.py`.
