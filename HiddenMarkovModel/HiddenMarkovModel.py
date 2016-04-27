import numpy as np
import math

class HMM:
    
    # transition matrix: P(state x to y) is a[x][y]
    # emission matrix: P(state x emit y) is e[x][y]
    
    def __init__(self, num_states, transition_probs, emission_probs, start_probs):
        self.num_states = num_states
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.emission_symbols = emission_probs[0].size
        self.starts = start_probs

    def forward(self, seq, start = None, trans = None, emiss = None):
        if start is None:
            start = self.starts
            trans = self.transitions
            emiss = self.emissions
        starts = start
        transitions = trans
        emissions = emiss
        T = seq.size

        forward_matrix = np.zeros((self.num_states, seq.size))

        scaling_factors = np.empty(T)
        log_prob = 0
        
        forward_matrix[:,0] = starts[:] * emissions[:,seq[0]]
        
        normalising_factor_first_column = np.sum(forward_matrix[:,0])
        if normalising_factor_first_column == 0:
            normalising_factor_first_column = 1
        else:
            normalising_factor_first_column = 1/normalising_factor_first_column
        forward_matrix[:,0] *= normalising_factor_first_column
        
        scaling_factors[0] = normalising_factor_first_column
        log_prob += math.log(normalising_factor_first_column)
        
        #Forward pass
        for symbol_index, symbol in enumerate(seq[1:], start=1):

            forward_matrix[:,symbol_index] = emissions[:,symbol] * np.dot(forward_matrix[:,symbol_index-1], transitions)

            # Normalise forward matrix
            forward_normalising_factor = np.sum(forward_matrix[:,symbol_index])
            # Check for this case where the sum is 0
            if forward_normalising_factor==0:
                forward_normalising_factor=1
            else:
                forward_normalising_factor = 1/forward_normalising_factor
                
            forward_matrix[:,symbol_index] *= forward_normalising_factor
            scaling_factors[symbol_index] = forward_normalising_factor
            log_prob += math.log(forward_normalising_factor)

        return forward_matrix, scaling_factors, log_prob

    def backward(self, seq, scaling_factors, start = None, trans = None, emiss = None):
        if start is None:
            start = self.starts
            trans = self.transitions
            emiss = self.emissions
        starts = start
        transitions = trans
        emissions = emiss
        T = seq.size

        backward_matrix = np.zeros((self.num_states, seq.size))

        #Backward pass
        backward_matrix[:,T-1] = 1/scaling_factors[T-1]
            
        for t in range(T-2,-1,-1):
            backward_matrix[:,t] = np.dot(transitions, emissions[:,seq[t+1]]*backward_matrix[:,t+1])
            # Normalise backward matrix using the same scaling factors as for the forward matrix
            backward_matrix[:,t] *= scaling_factors[t]

        return backward_matrix
                
    
    def sequence_log_probability(self, seq):
        # http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf for log probability calculations
        
        _, _, log_prob = self.forward(seq)
        
        return -log_prob
    
    def learn(self, seq, max_iters=100, threshold = 0.001, restart_threshold = 0.1, max_restarts = 10, inertia=0):
        # http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf for probability calculations
        # http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf for scaling and log probabilities
        
        transitions = self.transitions
        emissions = self.emissions
        starts = self.starts
        
        best_transitions = transitions
        best_emissions = emissions
        best_starts = starts
        best_score = -1e300
               
        num_restarts = 0
        
        T = seq.size
        
        forward_matrix = np.zeros((self.num_states, seq.size))
        backward_matrix = np.zeros((self.num_states, seq.size))
        
        done = False
        iters = 1
        old_score = -1e300
        
        while(not done):
            
            forward_matrix, scaling_factors, log_prob = self.forward(seq, start = starts, trans=transitions, emiss= emissions)
                
            backward_matrix = self.backward(seq, scaling_factors, start = starts, trans=transitions, emiss= emissions)

            #Re-estimate the parameters
            new_transitions = np.empty((self.num_states, self.num_states))
            new_emissions = np.empty((self.num_states, self.emission_symbols))
            new_starts = np.empty(self.num_states)

            #Re-estimate start probabilities
            new_starts = forward_matrix[:,0] * backward_matrix[:,0] / (scaling_factors[0]**2)
                
            # Re-estimate transitions
            for (i,j) in np.ndindex(self.num_states, self.num_states):
                
                temp_sum_num = np.sum(forward_matrix[i,:-1] * backward_matrix[j,1:] * emissions[j,seq[1:]])
                temp_sum_num *= transitions[i,j]
                
                temp_sum_denom = np.sum(forward_matrix[i,:]*backward_matrix[i,:] / scaling_factors)
                    
                #print("Transition: ",temp_sum_num," / ", temp_sum_denom)
                new_transitions[i,j] = temp_sum_num / temp_sum_denom

            #Re-estimate emissions
            for (j,k) in np.ndindex(self.num_states, self.emission_symbols):
                temp = forward_matrix[j,:]*backward_matrix[j,:] / scaling_factors
                temp_sum_num = np.sum(temp[ seq == k ])
                temp_sum_denom = np.sum(temp)
                
                #print("Emission: ",temp_sum_num," / ", temp_sum_denom)
                new_emissions[j,k] = temp_sum_num / temp_sum_denom

            new_score = -log_prob
            
            #Normalise the rows of the matrices
            new_starts /= np.sum(new_starts)
            new_transitions /= np.sum(new_transitions, axis=1)[:,np.newaxis]
            new_emissions /= np.sum(new_emissions, axis=1)[:,np.newaxis]
            
            # Check if we need to update our best parameters so far
            if new_score > best_score:
                best_transitions = transitions
                best_emissions = emissions
                best_starts = starts
                best_score = new_score
                
                
            # If we have reached the maximum iterations then we are done
            if iters >= max_iters:           
                done = True
            else:
                
                #If we have gotten better, but the amount is below the threshold, then we consider ourselves converged
                if new_score > old_score and new_score-old_score < threshold:
                    done = True
                    
                #If we are within the restart threshold, or have gotten worse, restart
                elif new_score - old_score < restart_threshold:                   
                    
                    if num_restarts > max_restarts:
                        #If we have gotten worse then we are done, else wait for convergence
                        if new_score < old_score:
                            done = True
                            
                    elif num_restarts == max_restarts:
                        #We have already made the max number of restarts
                        num_restarts += 1
                        #Go back to the best so far, and then wait for convergence from there
                        transitions = best_transitions
                        emissions = best_emissions
                        starts = best_starts
                        
                    # Inject some randomness into the parameters to break out of the local max
                    else:
                        num_restarts+=1
                        random_scaling = 1
                        transitions = np.random.rand(self.num_states,self.num_states) * random_scaling
                        transitions /= transitions.sum(axis=1)[:,np.newaxis]
                        starts = np.random.rand(self.num_states) * random_scaling
                        starts /= np.sum(starts)
                        emissions = np.random.rand(self.num_states,self.emission_symbols) * random_scaling
                        emissions /= emissions.sum(axis=1)[:,np.newaxis]
                        old_score = -1e300
                        
                #Iterating normally
                else:
                    old_score = new_score
                    transitions = new_transitions
                    emissions = new_emissions
                    starts = new_starts
                
            iters += 1
            
        # Set the models parameters to be the best estimated ones
        self.transitions = inertia*self.transitions + (1-inertia)*best_transitions
        self.emissions = inertia*self.emissions + (1-inertia)*best_emissions
        self.starts = inertia*self.starts + (1-inertia)*best_starts