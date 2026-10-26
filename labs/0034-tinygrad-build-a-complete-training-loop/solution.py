from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a tinygrad model and return per-epoch metrics.

    Steps:
    1. Create optimizer: Adam(get_parameters(model), lr=lr).
    2. For each epoch:
       a. Shuffle training data (Tensor.randperm or numpy shuffle).
       b. Iterate mini-batches: zero_grad -> forward -> loss -> backward -> step.
       c. Set Tensor.training = False, evaluate on val set.
       d. Append {'epoch', 'train_loss', 'val_loss', 'val_accuracy'} to history.
    3. Return the history list.
    """
    history = []
    # TODO: implement the training loop
    optimizer = Adam(get_parameters(model), lr=lr)
    n, m = X_train.shape
    Tensor.training = True

    for epoch in range(epochs):
      Tensor.training = True
      idx = Tensor.randperm(n)
      x_shuffled, y_shuffled = X_train[idx], y_train[idx]
      total_train_loss = 0.0
      n_batches = 0

      for start in range(0, n, batch_size):
         x_batch = x_shuffled[start:start+batch_size]
         y_batch = y_shuffled[start:start+batch_size]
         optimizer.zero_grad()

         y_pred = model(x_batch)
         loss = y_pred.sparse_categorical_crossentropy(y_batch)
         loss.backward()
         optimizer.step()

         total_train_loss += loss.item() 
         n_batches += 1

      train_loss = total_train_loss / n_batches

      Tensor.training = False
      y_pred_val = model(X_val)
      val_loss = y_pred_val.sparse_categorical_crossentropy(y_val).item()
      val_accuracy = (y_pred_val.argmax(axis=1) == y_val).mean().item()

      result = {
         'epoch': epoch+1,
         'train_loss': train_loss,
         'val_loss': val_loss,
         'val_accuracy': val_accuracy
      } 

      history.append(result)

    return history
