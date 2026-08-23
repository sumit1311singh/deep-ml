def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	# P(B) = P(A,B) + P(`A,B)
	p_B = joint_distribution[('A','B')] + joint_distribution[('`A','B')]
	
	if p_B == 0:
		return 0.0

    # P(A|B) = P(A,B) / P(B)
	return joint_distribution[('A','B')] / p_B