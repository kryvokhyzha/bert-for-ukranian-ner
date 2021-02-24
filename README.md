# bert-for-ukranian-ner
This repository contains I part of my bachelor's project


## How to setup an environment?

This template use `poetry` to manage dependencies of your project.

First you need to [install poetry](https://python-poetry.org/docs/#installation).

Then if you use `conda` (recommended) to manage environments (to use regular virtualenvenv just skip this step):

* tell `poetry` not to create new virtualenv for you

    (instead `poetry` will use currently activated virtualenv):

    `poetry config virtualenvs.create false`

* create new `conda` environment for your project (change env name for your desired one):

    `conda create -n bert_for_ner python=3.8`

* actiave environment:

    `conda activate bert_for_ner`

Now you are ready to add dependencies to your project. For this use [`add` command](https://python-poetry.org/docs/cli/#add):

`poetry add scikit-learn torch <any_package_you_need>`

Next run `poetry install` to check your final state are even with configs.

After that add changes to git and commit them `git add pyproject.toml poetry.lock`

Finally add `pre-commit` hooks to git: `pre-commit install`
