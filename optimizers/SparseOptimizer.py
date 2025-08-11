import torch
import torch.optim as optim
import numpy as np

class SO(optim.Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.4, 0.999), eps=1e-8, weight_decay=0, density_ratio=0.1, T=30):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.density_ratio = density_ratio  # Percentage of parameters to select for sparsification
        self.T = T
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SO, self).__init__(params, defaults)
        
    def _select_sparse_indices(self, grad_matrix, num_non_zero, state):
        """
        Converts a dense matrix to a sparse matrix based on randomness.
        Uses the same selection for T iterations, then updates.
        """
        size = grad_matrix.shape
        if state['iteration_count'] == self.T or state['step'] == 0:
            # Select random gradients
            random_indices = torch.randperm(grad_matrix.numel(), device=grad_matrix.device)[:num_non_zero]
            state['selected_indices'] = random_indices

            # Reset iteration count for this layer and store action
            state['iteration_count'] = 1
            
        else:
            # Increment iteration count without changing indices
            state['iteration_count'] += 1

        # Convert 1D indices back into 2D indices for the matrix
        indices = torch.stack([state['selected_indices'] // size[1], state['selected_indices'] % size[1]])

        # Create a sparse tensor from the selected gradients
        values = grad_matrix.view(-1)[state['selected_indices']]
        sparse_grad = torch.sparse_coo_tensor(indices, values, size, device=grad_matrix.device)
        
        return sparse_grad
        
    def sparse_update(self, grad_sparse, exp_avg_values, exp_avg_sq_values, exp_avg_indices, beta1, beta2, num_non_zero, lr, step, eps):

        device = grad_sparse.device
        
        # Get gradient indices and values
        grad_indices = grad_sparse._indices()
        grad_values = grad_sparse._values()
        
        # Create bias correction variables
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
            
        if step > 1:
            # Create a mask for overlapping values (intersection of indices between grad and exp_avg)
            grad_exp_avg_mask = (grad_indices.unsqueeze(2) == exp_avg_indices[:, :num_non_zero].unsqueeze(1)).all(dim=0)
            overlap_mask_grad = grad_exp_avg_mask.any(dim=1)  # Mask for grad_indices that overlap with exp_avg_indices
            new_grad_mask = ~overlap_mask_grad  # Mask for grad_indices that don't overlap (new gradients)  
            overlap_mask_exp_avg = grad_exp_avg_mask.any(dim=0) # Mask for exp_avg_indices that overlap with grad_indices
            new_mask_exp_avg = ~overlap_mask_exp_avg
                    
            # Add num_current_elements of False at the end of overlap_mask_grad
            overlap_mask_exp_avg = torch.cat([overlap_mask_exp_avg, torch.zeros(num_non_zero, dtype=torch.bool, device=device)])
            new_mask_exp_avg = torch.cat([new_mask_exp_avg, torch.zeros(num_non_zero, dtype=torch.bool, device=device)])
            
            # In-place update of overlapping values for exp_avg and exp_avg_sq using index_put_
            nb_overlap = overlap_mask_grad.sum()
                
            if nb_overlap > 0:
            
                if nb_overlap > 1:
                    # Compute exp_overlap_indices 
                    exp_overlap_indices = exp_avg_indices[:, overlap_mask_exp_avg]
                    
                    # Use the new lexsort_indices function to reorder exp_avg_indices in-place
                    sort_order = lexsort_indices(exp_overlap_indices)
                    
                    # Update exp_avg_values using in-place indexed assignment
                    sorted_exp_avg_values = exp_avg_values[overlap_mask_exp_avg][sort_order]
                    exp_avg_values.index_put_((overlap_mask_exp_avg,), sorted_exp_avg_values)
                    
                    del sorted_exp_avg_values
                    
                    # Update exp_avg_sq_values using in-place indexed assignment
                    sorted_exp_avg_sq_values = exp_avg_sq_values[overlap_mask_exp_avg][sort_order]
                    exp_avg_sq_values.index_put_((overlap_mask_exp_avg,), sorted_exp_avg_sq_values)
                    
                    del sorted_exp_avg_sq_values
                    
                    # Update exp_avg_indices using in-place indexed assignment
                    sorted_indices = exp_overlap_indices[:, sort_order]
                    first_dim_indices = torch.arange(exp_avg_indices.size(0), device=device).unsqueeze(1).expand(-1, nb_overlap)
                    exp_avg_indices.index_put_((first_dim_indices, overlap_mask_exp_avg), sorted_indices)
                    
                    del first_dim_indices, sorted_indices, sort_order, exp_overlap_indices
                    
                    # Compute grad_overlap_indices
                    grad_overlap_indices = grad_indices[:, overlap_mask_grad]
                    
                    # Use lexsort_indices to sort overlapping grad_indices
                    sort_order = lexsort_indices(grad_overlap_indices)

                    # Update grad_values in sorted order for overlap
                    sorted_grad_values = grad_values[overlap_mask_grad][sort_order]
                    grad_values.index_put_((overlap_mask_grad,), sorted_grad_values)
                    
                    del sorted_grad_values
                    
                    # Update exp_avg_sq_values using in-place indexed assignment
                    sorted_indices = grad_overlap_indices[:, sort_order]
                    first_dim_indices = torch.arange(grad_indices.size(0), device=device).unsqueeze(1).expand(-1, nb_overlap)
                    grad_indices.index_put_((first_dim_indices, overlap_mask_grad), sorted_indices)
                    
                    del first_dim_indices, sorted_indices, sort_order, grad_overlap_indices
                
                # Update exp_avg and exp_avg_sq values in-place for overlapping indices
                exp_avg_values.index_put_((overlap_mask_exp_avg,), exp_avg_values[overlap_mask_exp_avg].mul_(beta1).add_(grad_values[overlap_mask_grad], alpha=1 - beta1))
                exp_avg_sq_values.index_put_((overlap_mask_exp_avg,), exp_avg_sq_values[overlap_mask_exp_avg].mul_(beta2).addcmul_(grad_values[overlap_mask_grad], grad_values[overlap_mask_grad], value=1 - beta2))
            
            del overlap_mask_exp_avg, overlap_mask_grad, grad_exp_avg_mask    
                
            if new_mask_exp_avg.any():
                exp_avg_values.index_put_((new_mask_exp_avg,), exp_avg_values[new_mask_exp_avg].mul_(beta1))
                exp_avg_sq_values.index_put_((new_mask_exp_avg,), exp_avg_sq_values[new_mask_exp_avg].mul_(beta2))
                
            del new_mask_exp_avg
            
            # Add new gradient indices and values to exp_avg and exp_avg_sq
            if new_grad_mask.any():
                nb_new_grad = num_non_zero + new_grad_mask.sum().item()                
                indices = torch.arange(num_non_zero, nb_new_grad, device=device)
                exp_avg_values.index_put_((indices,), grad_values[new_grad_mask].mul(1 - beta1))
                exp_avg_sq_values.index_put_((indices,), grad_values[new_grad_mask].pow(2).mul_(1 - beta2))
                first_dim_indices = torch.arange(exp_avg_indices.size(0), device=device).unsqueeze(1).expand(-1, indices.size(0))
                exp_avg_indices.index_put_((first_dim_indices, indices), grad_indices[:, new_grad_mask])
                
                del new_grad_mask
                
                # Update parameters 
                update = torch.sparse_coo_tensor(exp_avg_indices[:,:nb_new_grad].clone(), - step_size * exp_avg_values[:nb_new_grad].div(exp_avg_sq_values[:nb_new_grad].div(bias_correction2).sqrt_().add_(eps)), grad_sparse.size(), device=device)

                # Select the top num_non_zero elements by magnitude
                _, top_indices = torch.topk(torch.abs(exp_avg_values[:nb_new_grad]), num_non_zero)
                
                # Update exp_avg_values and exp_avg_sq_values in-place with top values
                indices = torch.arange(0, num_non_zero, device=device)
                copy_1 = exp_avg_values[top_indices]
                exp_avg_values.index_put_((indices,), copy_1)
                copy_2 = exp_avg_sq_values[top_indices]
                exp_avg_sq_values.index_put_((indices,), copy_2)

                del copy_1, copy_2
                        
                # Use indexing to update all elements in the first dimension
                first_dim_indices = torch.arange(exp_avg_indices.size(0), device=device).unsqueeze(1).expand(-1, indices.size(0))
                copy_3 = exp_avg_indices[:, top_indices]
                exp_avg_indices.index_put_((first_dim_indices, indices), copy_3)
                    
                del first_dim_indices, indices, top_indices, copy_3
                
            else:
                # Update parameters 
                update = torch.sparse_coo_tensor(exp_avg_indices[:,:num_non_zero].clone(), - step_size * exp_avg_values[:num_non_zero].div(exp_avg_sq_values[:num_non_zero].div(bias_correction2).sqrt_().add_(eps)), grad_sparse.size(), device=device)
        else :
            indices = torch.arange(0, num_non_zero, device=device)
            exp_avg_values.index_put_((indices,), grad_values).mul_(1 - beta1)
            exp_avg_sq_values.index_put_((indices,), grad_values.pow(2)).mul_(1 - beta2)
            first_dim_indices = torch.arange(exp_avg_indices.size(0), device=device).unsqueeze(1).expand(-1, indices.size(0))
            exp_avg_indices.index_put_((first_dim_indices, indices), grad_indices)
                
            del first_dim_indices, indices
            
            # Update parameters 
            update = torch.sparse_coo_tensor(exp_avg_indices[:,:num_non_zero].clone(), - step_size * exp_avg_values[:num_non_zero].div(exp_avg_sq_values[:num_non_zero].div(bias_correction2).sqrt_().add_(eps)), grad_sparse.size(), device=device)
                
        return update
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            is_dense_layer = group['dense']
            
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SRO does not support pre-existing sparse gradients.')

                # State initialization
                state = self.state[param]
                beta1, beta2 = group['betas']
                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']

                # Dense Operations
                if is_dense_layer:
                    
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(param.data)
                        state['exp_avg_sq'] = torch.zeros_like(param.data)
                    
                    state['step'] += 1
                    
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    # Apply weight decay if needed
                    if weight_decay != 0:
                        grad.add_(param.data, alpha=weight_decay)
                    
                    # Update first moment (exp_avg) and second moment (exp_avg_sq)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias correction for first and second moment estimates
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Bias-corrected moments
                    exp_avg_hat = exp_avg / bias_correction1
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2

                    # Compute the final update step for parameters
                    exp_avg_sq_hat.sqrt_().add_(eps)
                    step_size = lr / bias_correction1

                    # Apply the final update to the parameter
                    param.data.addcdiv_(exp_avg_hat, exp_avg_sq_hat, value=-step_size)
                    
                    # Delete the gradient after the update
                    param.grad = None
                    
                    # Delete temporary variables to free memory
                    del exp_avg_hat
                    del exp_avg_sq_hat 

                # Sparse Operations
                else:
                    device = grad.device
                    
                    # Apply weight decay if needed
                    if weight_decay != 0:
                        grad.add_(param.data, alpha=weight_decay)
                        
                    # Compute the number of non-zero entries based on sparsity ratio
                    num_weights = np.prod(grad.shape)
                    num_non_zero = int(num_weights * self.density_ratio)

                    # Initialize state-specific values for this layer if not done
                    if 'selected_indices' not in state:
                        state['selected_indices'] = None
                        state['iteration_count'] = 0
                        state['step'] = 0
                        
                    # Convert the gradient to sparse based on dynamic balance between randomness and importance
                    sparse_grad = self._select_sparse_indices(grad, num_non_zero, state)
                    
                    # Delete the gradient after the update
                    param.grad = None

                    if state['step'] == 0:
                        
                        # Calculate the maximum size (double the current number of elements in sparse_grad)
                        max_num_elements = num_non_zero * 2  # Double the number of elements

                        # Generate sequential indices using division and modulo
                        row_indices = torch.arange(max_num_elements, device=device) // sparse_grad.size(1)
                        col_indices = torch.arange(max_num_elements, device=device) % sparse_grad.size(1)

                        # Combine the row and column indices into the final tensor of shape (2, max_num_elements)
                        state['exp_avg_indices'] = torch.stack([row_indices, col_indices], dim=0)

                        # Preallocate memory for exp_avg and exp_avg_sq with double the capacity
                        state['exp_avg_values'] = torch.zeros(max_num_elements, dtype=sparse_grad._values().dtype, device=device)
                        state['exp_avg_sq_values'] = torch.zeros(max_num_elements, dtype=sparse_grad._values().dtype, device=device)

                        del row_indices, col_indices
                        
                    state['step'] += 1

                    # Update sparse moments 
                    update = self.sparse_update(sparse_grad, state['exp_avg_values'], state['exp_avg_sq_values'], state['exp_avg_indices'], beta1, beta2, num_non_zero, lr, state['step'], eps)

                    # Delete the sparse_grad after the moments have been updated
                    del sparse_grad

                    # Apply the sparse update
                    param.data.add_(update)
                    
                    del update
                        
def lexsort_indices(indices):
    """
    Sort a 2D indices tensor by rows and then columns, similar to numpy.lexsort.
    
    Parameters:
    - indices (torch.Tensor): A 2D tensor of shape (2, N) where N is the number of indices.
    
    Returns:
    - torch.Tensor: The sort order tensor for reordering indices.
    """
    # Sort by columns first, then by rows to achieve lexicographic order
    _, col_order = indices[1].sort()
    sorted_row_indices = indices[0][col_order]
    _, final_sort_order = sorted_row_indices.sort()
    return col_order[final_sort_order]

