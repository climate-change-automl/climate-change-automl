<br/>
<p align="center"><img src="img/ccautoml_logo.png" width=700 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

# A Call to Action

Climate change is one of the most pressing issues facing humanity, and although automated machine learning (AutoML) techniques have been developing rapidly, they do not yet perform well out of the box for high-impact climate change datasets.
We give a `call to action` to the AutoML community to use AutoML methods on challenging CCAI datasets.
In this repository, we run popular AutoML libraries on datasets for climate modeling, catalyst prediciton, and wind power forecasting.

# Table of contents
1. [ClimART](#ClimART)
2. [Open Catalyst Project](#OpenCatalystProject)
3. [SDWPF](#SDWPF)
4. [CCAI Resources](#CCAIResources)
5. [AutoML Resources](AutoMLResources)

# ClimART <a name="ClimART"></a>

Numerical weather prediction models, as well as global and regional climate models, give crucial information to policymakers and the public about the impact of changes in the Earth's climate.
The bottleneck is atmospheric radiative transfer (ART) calculations, which are used to compute the heating rate of any given layer of the atmosphere.
While ART has historically been calculated using computationally intensive physics simulations, researchers have recently used neural networks to substantially reduce the computational bottleneck, enabling ART to be run at finer resolutions and obtaining better overall predictions.

We use the [ClimART dataset](https://github.com/RolnickLab/climart) from the [NeurIPS Datasets and Benchmarks Track 2021](https://openreview.net/forum?id=FZBtIpEAb5J). 
It consists of global snapshots of the atmosphere across a discretization of latitude, longitude, atmospheric height, and time from 1979 to 2014.
Each datapoint contains measurements such as temperature, water vapor, and aerosols at different atmospheric heights.
As with prior work, we use MLPs, CNNs, GNNs, and GCNs as baselines.

# Open Catalyst Project <a name="OpenCatalystProject"></a>

Discovering new catalysts is key to cost-effective chemical reactions to address the problem of energy storage, which is necessitated by the intermittency of power generation from growing renewable sources, such as wind and solar. 
Catalyst discovery is also important for more efficient production of ammonia fertilizer, which currently makes up 1\% of the world's CO$_2$ emissions.
Modern design of catalysts uses a simulation via density functional theory (DFT), which can be approximated using deep learning.
Specifically, given a set of atomic positions for the reactants and catalyst, the energy of the structure can be predicted.

We use the [Open Catalyst 2020 (OC20) dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md), which was featured in a [NeurIPS 2021 contest](https://opencatalystproject.org/challenge.html).
Each datapoint is one reaction, where the features consist of the initial starting positions of the atoms, and the label consists of the energy needed to drive the reaction.
As a baseline, we use [Graphormer](https://github.com/microsoft/Graphormer), the winning solution from the NeurIPS 2021 Open Catalyst Challenge, developed by Microsoft.

# SDWPF <a name="SDWPF"></a>

Wind power is one of the leading renewable energy types, since it is cheap, efficient, and harmless to the environment.
The only major downside in wind power is its unreliablility: changes in wind speed and direction make the energy gained from wind power inconsistent.
In order to keep the balance of energy generation and consumption on the power grid, other sources of energy must be added on short notice when wind power is down, which is not always possible (for example, coal plants take at least 6 hours to start up).
Therefore, forecasting wind power is an important problem that must be solved to facilitate greater adoption of wind power.

We use the [SDWPF (Spatial Dynamic Wind Power Forecasting) dataset](https://arxiv.org/abs/2208.04360), which was recently featured in a [KDD Cup 2022 competition](https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets) that included 2490 participants.
This is by far the largest wind power forecasting dataset, consisting of data from 134 wind turbines across 12 months.
The features consist of external features such as wind speed, wind direction, and temperature, and turbine features such as pitch angle of the blades, operating status, relative location, and elevation.
As baselines, we use a [BERT-based model](https://github.com/LongxingTan/KDDCup2022-Baidu) and a [GRU+LGBM model](https://github.com/linfangquan/kddcup2022), which placed 3rd and 7th (1st and 3rd among open-source models), respectively. 

# CCAI Resources <a name="CCAIResources"></a>

 - [Climate Change AI website](https://www.climatechange.ai/)
 - [CCAI Wiki](https://wiki.climatechange.ai/wiki/Welcome_to_the_Climate_Change_AI_Wiki)
 - [NeurIPS 2022 Workshop](https://www.climatechange.ai/events/neurips2022)
 - [Tackling Climate Change with Machine Learning](https://dl.acm.org/doi/10.1145/3485128)
 - [Awesome CCAI :sunglasses:](https://github.com/shankarj67/Awesome-Climate-Change-AI)

# AutoML Resources <a name="AutoMLResources"></a>

 - [AutoML.org](http://automl.org/)
 - [AutoML Conference](https://automl.cc/)
 - [AutoML book](https://www.automl.org/book/)
 - [Awesome AutoML :sunglasses:](https://github.com/hibayesian/awesome-automl-papers)
 - [Awesome AutoDL :sunglasses:](https://github.com/D-X-Y/Awesome-AutoDL)


