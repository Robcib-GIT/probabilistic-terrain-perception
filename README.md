# probabilistic-terrain-perception
Terrain evaluation using CNN to generate a discretized matrix of probabilities with zones related to stability and potential support points for legged robots.
Both terrain (gravel, dirt road, etc.) and obstacles are characterized probabilistically (0 - 100% probability of overcoming) generating a gradient in their surrounding areas. This information serves as a basis for making decisions regarding the selection of support points for legged robots.

Processed and original terrain:

![Probabilistic terrain](https://github.com/Robcib-GIT/probabilistic-terrain-perception/assets/57187750/19b139b2-110a-4b89-aa89-ee31079a98a0)
