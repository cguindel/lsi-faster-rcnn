import numpy as np
import math

def walpha_angle(probs, viewp_bins, viewp_offset):
    max_bin = np.argmax(probs)
    assert max_bin < viewp_bins
    prob_max_bin = probs[max_bin]
    max_bin_minus = max_bin-1 if max_bin>0 else viewp_bins-1
    max_bin_plus = max_bin+1 if max_bin<viewp_bins-1 else 0
    if probs[max_bin_minus] > probs[max_bin_plus]:
        prob_contig = probs[max_bin_minus]
        contig_bin = max_bin_minus
    else:
        prob_contig = probs[max_bin_plus]
        contig_bin = max_bin_plus

    norm_prob_max_bin = prob_max_bin / (prob_max_bin+prob_contig)
    norm_prob_contig = prob_contig / (prob_max_bin+prob_contig)

    max_alpha = math.pi * (2 * max_bin + 1)/viewp_bins - viewp_offset
    contig_alpha = math.pi * (2 * contig_bin + 1)/viewp_bins - viewp_offset

    if abs(max_alpha-contig_alpha) > 2*math.pi - math.pi/4 - 0.01:
        if max_alpha-contig_alpha > 0:
            max_alpha -= 2*math.pi
        else:
            contig_alpha -= 2*math.pi

    estimated_angle = max_alpha*norm_prob_max_bin + contig_alpha*norm_prob_contig
    if estimated_angle < 0:
        estimated_angle += 2*math.pi

    if estimated_angle > math.pi:
        estimated_angle = estimated_angle - 2*math.pi

    return estimated_angle

def bin_center_angle(probs, viewp_bins, viewp_offset):
    max_bin = np.argmax(probs)
    assert max_bin < viewp_bins
    estimated_angle = math.pi * (2 * max_bin + 1)/viewp_bins \
                        - viewp_offset
    if estimated_angle > math.pi:
        estimated_angle = estimated_angle - 2*math.pi

    return estimated_angle

def kl_angle(probs, viewp_bins, viewp_offset):
    i_star_list = []
    min_kl_list = []
    for vbin in xrange(0, viewp_bins-1):
        L = 2*math.pi/viewp_bins
        theta_i = math.pi * (2 * vbin + 1)/viewp_bins \
                            - viewp_offset
        if vbin+1 > viewp_bins-1:
            theta_i_plus_one = math.pi /viewp_bins \
                                - viewp_offset
            theta_i_star = theta_i + (probs[0] * L)/(probs[vbin]+probs[0])
            min_kl_distance = ((theta_i_plus_one-theta_i_star)/L) \
                                * math.log((theta_i_plus_one-theta_i_star)/(L*probs[vbin])) \
                                + ((theta_i_star-theta_i)/L) * math.log((theta_i_star-theta_i)/(L*probs[vbin+1]))
        else:
            theta_i_plus_one = math.pi * (2 * (vbin+1) + 1)/viewp_bins \
                                - viewp_offset
            theta_i_star = theta_i + (probs[vbin+1] * L)/(probs[vbin]+probs[vbin+1])

            if (theta_i_plus_one-theta_i_star)/(L*probs[vbin]) <= 0:
                min_kl_distance = ((theta_i_star-theta_i)/L) * math.log((theta_i_star-theta_i)/(L*probs[0]))
            elif (theta_i_star-theta_i)/(L*probs[0]) <= 0:
                min_kl_distance = ((theta_i_plus_one-theta_i_star)/L) \
                                    * math.log((theta_i_plus_one-theta_i_star)/(L*probs[vbin]))
            else:
                min_kl_distance = ((theta_i_plus_one-theta_i_star)/L) \
                                    * math.log((theta_i_plus_one-theta_i_star)/(L*probs[vbin])) \
                                    + ((theta_i_star-theta_i)/L) * math.log((theta_i_star-theta_i)/(L*probs[0]))

        min_kl_list.append(min_kl_distance)
        i_star_list.append(theta_i_star)
    argmin_kls = np.argmin(np.array(min_kl_list))
    estimated_angle = i_star_list[argmin_kls]

    return estimated_angle
