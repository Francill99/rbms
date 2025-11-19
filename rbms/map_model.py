from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.ising.ising_rbm import IsingRBM
from rbms.classes import RBM, EBM
from rbms.potts_bernoulli.classes import PBRBM
from rbms.bernoulli_gaussian.classes import BGRBM

map_model: dict[str, EBM] = {"BBRBM": BBRBM, "PBRBM": PBRBM, "BGRBM": BGRBM}
