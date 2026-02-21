import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        T = len(input_observation_states)
        N = len(self.hidden_states)

        if T == 0:
            raise ValueError("Observation seq input is emp.")

        # convert obs to id
        Obs_Idx = np.zeros(T, dtype=int)
        for t in range(T):
            Obs_Idx[t] = self.observation_states_dict[input_observation_states[t]]

        #Scaling or underflow!
        Forward_Table = np.zeros((T, N), dtype=float)
        Scale_Factors = np.zeros(T, dtype=float)
        
        # Step 2. Calculate probabilities
        for i in range(N):
            Forward_Table[0, i] = self.prior_p[i] * self.emission_p[i, Obs_Idx[0]]
        #need to scale first!!!
        Scale_Factors[0] = np.sum(Forward_Table[0, :])
        Forward_Table[0, :] = Forward_Table[0, :] / Scale_Factors[0]
        #recurrsion 
        for t in range(1, T):
            for i in range(N):
                Total = 0.0
                for j in range(N):
                    Total += Forward_Table[t - 1, j] * self.transition_p[j, i]
                Forward_Table[t, i] = self.emission_p[i, Obs_Idx[t]] * Total

            Scale_Factors[t] = np.sum(Forward_Table[t, :])
            Forward_Table[t, :] = Forward_Table[t, :] / Scale_Factors[t]


        # Step 3. Return final probability 
        Forward_Probability = 1.0
        for t in range(T):
            Forward_Probability *= Scale_Factors[t]

        return float(Forward_Probability)


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        


        # Step 1. Initialize variables
   
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))  
        
        T = len(decode_observation_states)
        N = len(self.hidden_states)

        Obs_Idx = np.zeros(T, dtype=int)
        for t in range(T):
            Obs_Idx[t] = self.observation_states_dict[decode_observation_states[t]]

        Backpointer = np.zeros((T, N), dtype=int)
       
        # Step 2. Calculate Probabilities
        
        for i in range(N):
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, Obs_Idx[0]]
            Backpointer[0, i] = 0

        # Recursion X(
        for t in range(1, T):
            for i in range(N):
                Best_Prev = 0
                Best_Val = -1.0
                for j in range(N):
                    Val = viterbi_table[t - 1, j] * self.transition_p[j, i]
                    if Val > Best_Val:
                        Best_Val = Val
                        Best_Prev = j

                viterbi_table[t, i] = self.emission_p[i, Obs_Idx[t]] * Best_Val
                Backpointer[t, i] = Best_Prev

            
        # Step 3. Traceback 
        #start wuth best final state
        Last_State = int(np.argmax(viterbi_table[T - 1, :]))
        best_path[T - 1] = Last_State

        for t in range(T - 2, -1, -1):
            best_path[t] = Backpointer[t + 1, int(best_path[t + 1])]

        # Step 4. Return best hidden state sequence 
        Best_Hidden_State_Sequence = []
        for t in range(T):
            Idx = int(best_path[t])
            Best_Hidden_State_Sequence.append(self.hidden_states_dict[Idx])

        return Best_Hidden_State_Sequence
        