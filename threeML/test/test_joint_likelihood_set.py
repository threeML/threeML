from threeML import *
from conftest import data_list_bn090217206_nai6, get_grb_model


# Define two dummy functions to return always the same model and the same
# dataset
def get_model(id):

    return get_grb_model(Powerlaw())


def get_data(id):

    return data_list_bn090217206_nai6()


def test_joint_likelihood_set():

    jlset = JointLikelihoodSet(data_getter=get_data, model_getter=get_model, n_iterations=10)

    jlset.go(compute_covariance=False)


def test_joint_likelihood_set_parallel():

    jlset = JointLikelihoodSet(data_getter=get_data, model_getter=get_model, n_iterations=10)

    with parallel_computation(start_cluster=False):

        res = jlset.go(compute_covariance=False)

    print(res)


