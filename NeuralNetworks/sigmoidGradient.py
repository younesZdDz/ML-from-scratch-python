from scipy.special import expit


def sigmoidGradient(z):
    return(expit(z)*(1-expit(z)))