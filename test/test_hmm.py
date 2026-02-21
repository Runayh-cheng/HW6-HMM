import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    #forward
    model = HiddenMarkovModel(
        #weather
        mini_hmm['observation_states'],   
        #hot/cold
        mini_hmm['hidden_states'],       
        mini_hmm['prior_p'],         
        mini_hmm['transition_p'],
        mini_hmm['emission_p']
    )
    obs_seq = mini_input['observation_state_sequence']   
    expected_seq = list(mini_input['best_hidden_state_sequence'])

    forward_prob = model.forward(obs_seq)

    #probablity check range
    assert 0.0 < forward_prob <= 1.0
    #check is float
    assert isinstance(forward_prob, float)
    #check value; actual value calculated by ChatGPT
    assert np.abs(forward_prob-0.03506441162109376) < 0.001

    #viterbi
    predicted_seq = model.viterbi(obs_seq)
    #pred for every obj
    assert len(predicted_seq) == len(obs_seq)
    # the right states check 
    valid_hidden = set(mini_hmm['hidden_states'])
    for state in predicted_seq:
        assert state in valid_hidden
    #right seq actually
    assert predicted_seq == expected_seq

    #edge case 1: only one object so prob should be 1 since no split
    one_obs = np.array(['sunny'])
    one_fwd = model.forward(one_obs)
    one_vit = model.viterbi(one_obs)
    assert len(one_vit) == 1
    #sunny should be hot
    assert one_vit[0] == 'hot'

    #edge case 2: input is empty vec
    emp = np.array([])
    with pytest.raises(ValueError):
        _ = model.forward(emp)



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    model = HiddenMarkovModel(
        full_hmm["observation_states"],
        full_hmm["hidden_states"],
        full_hmm["prior_p"],
        full_hmm["transition_p"],
        full_hmm["emission_p"]
    )

    obs_seq = full_input['observation_state_sequence']        
    expected_seq = list(full_input['best_hidden_state_sequence'])
    forward_prob = model.forward(obs_seq)
    #prob calculated and right range 
    assert forward_prob is not None
    assert 0.0 < forward_prob <= 1.0
    
    #viterbi check 
    predicted_seq = model.viterbi(obs_seq)
    #same as before 
    assert predicted_seq is not None
    #right num of output 
    assert len(predicted_seq) == len(obs_seq)
    #right state
    valid_hidden = set(full_hmm['hidden_states'])
    for state in predicted_seq:
        assert state in valid_hidden
    #check against actual 
    assert predicted_seq == expected_seq