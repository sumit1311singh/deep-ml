import numpy as np

def distillation_loss(
	student_logits: np.ndarray,
	teacher_logits: np.ndarray,
	temperature: float = 1.0
) -> float:
	"""
	Compute knowledge distillation loss.
	
	L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss value
	"""
	# Your code here
	student_scaled = student_logits / temperature
	teacher_scaled = teacher_logits / temperature

	teacher_shifted = teacher_scaled - np.max(
        teacher_scaled, axis=-1, keepdims=True
    )
	log_teacher_probs = teacher_shifted - np.log(
        np.sum(np.exp(teacher_shifted), axis=-1, keepdims=True)
    )
	teacher_probs = np.exp(log_teacher_probs)

	student_shifted = student_scaled - np.max(
        student_scaled, axis=-1, keepdims=True
    )
	log_student_probs = student_shifted - np.log(
		np.sum(np.exp(student_shifted), axis=-1, keepdims=True)
	)
	
	kl_divergence = np.sum(
		teacher_probs * (log_teacher_probs - log_student_probs),
		axis=-1
	)
	
	loss = (temperature ** 2) * np.mean(kl_divergence)
	
	return float(loss)