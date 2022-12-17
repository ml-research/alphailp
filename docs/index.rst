.. alphailp documentation master file, created by
   sphinx-quickstart on Mon Dec  5 22:15:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to alphailp's documentation!
====================================
alphaILP is a neuro-symbolic framework that can learn generalized rules from complex visual scenes.
alphaILP learns to represent scenes as logic programs—intuitively, logical atoms correspond to objects, attributes, and relations, and clauses encode highlevel scene information. alphaILP has an end-to-end reasoning architecture
from visual inputs. Using it, alphaILP performs differentiable inductive logic programming on complex visual scenes, i.e., the logical rules are
learned by gradient descent.
`[GitHub] <https://github.com/ml-research/alphailp>`_

Introduction by Examples
========================

We provide an introduction by giving specific examples of use cases of alphaILP.

* :doc:`Building-a-Reasoner` : A brief introduction about how to build a differentiable reasoner

* :doc:`Building-a-Learner` : A brief introduction about how to perform rule learning from visual scenes using alphaILP.

* :doc:`Compositional-Test` : A demonstration of an use case of the differentiable reasoner as a compositional checker for industrial automation.





Acknowledgements
================
This project has been supported by `SPAICER (01MK20015E) <https://www.spaicer.de/en/>`_ , `TAILOR (952215) <https://tailor-network.eu/>`_, and `AICO <https://careers.aico.ai/>`_.

.. image:: _static/spaicer.png
   :height: 100
.. image:: _static/tailor.png
   :height: 100
.. image:: _static/aico.png
   :height: 100





.. toctree::
   :maxdepth: 4
   :caption: Contents:

   Building-a-Reasoner
   Building-a-Learner
   Compositional-Test
   architecture
   valuation
   mode-declaration
   acknowledgements
   src


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
