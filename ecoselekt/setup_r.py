from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr


def main():
    pandas2ri.activate()
    devtools = importr("devtools")
    # install development version of ScottKnottESD for non-parametric test support
    devtools.install_github("klainfo/ScottKnottESD", ref="development")
    importr("ScottKnottESD")
