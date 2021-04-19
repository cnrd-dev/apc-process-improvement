# Measuring process improvements of Advanced Process Control (APC)

A [blog post](https://medium.com/@coenraad-pretorius/measuring-process-improvements-with-advanced-process-control-32945f3ff2e5) on Medium gives an overview of the project and discusses the results.

## Motivation

Determine the benefits by implementing APC in a process plant.

Three key questions:

1. Did the plant adopt APC?
1. What are the process improvements relating to stability and throughput?
1. Does APC improve energy performance?

## File descriptions

- Process analysis notebook: Uses one minute process data used the determine APC utilisation and process improvements.
- Energy analysis notebooks: Uses daily data to build a regression model to predict expected energy consumption and compare with actual energy consumption.
- myLib: Custom package containing functions used for data analysis.

## How to run the notebooks

Dependencies and virtual environment details are located in the `Pipfile` which can be used with `pipenv`.

## License

GNU GPL v3

## Author

Analysis done by Coenraad Pretorius.

## Acknowledgement

Acknowledgement to Emre for the code on [diagnostic plots](https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/).
