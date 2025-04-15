# Cartographer (beta)
Spectral library predictor based off of Chronologer
early beta used with XXX

Cartographer has been verified to work under the following environment:
| Package | Version |
| --- | --- |
| Python | 3.10.6 |
| PyTorch | 1.21.1 |
| Numpy | 1.23.2 |
| Pandas | 1.4.3 |
| Scipy | 1.9.1 |
| Pyteomics | 4.5.5 |


# Demo

A small test file (fasta/Scer.fasta) with 3 test proteins is included to test a working environment. From the command line, a library can be generated running

> python src/Generate_library.py fasta/Scer.fasta test.dlib

where "test.dlib" can be replaced with the output file name of your choice
