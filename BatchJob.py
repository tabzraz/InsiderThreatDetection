import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import time
import json

# My implementation of a Hidden Markov Model
from HiddenMarkovModel import HiddenMarkovModel

#names = [ "r4.1/r41_features_simple.h5", "r4.1/r41_features_complex.h5", "r4.2/r42_features_simple.h5", "r4.2/r42_features_complex.h5" ]
names = [ "r42_features_complex.h5", "r42_features_complex.h5", "r42_features_complex.h5", "r42_features_complex.h5" ]
saves = [ "userScores_complex_r42_50_inertia_10.json","userScores_complex_r42_50_inertia_20.json","userScores_complex_r42_50_inertia_50.json","userScores_complex_r42_50_inertia_100.json" ]
states = [ 10,20,50,100 ]
inertias = [0.05]
#saves = [ "userScores_simple_r41_80_inertia.json", "userScores_complex_r41_80_inertia.json", "userScores_simple_r42_80_inertia.json", "userScores_complex_r42_80_inertia.json"]

filename = "/home/tabz/Documents/CMU_Dataset/r4.2/" + "r42_features_simple.h5" #+ name
# filename = "C:/Users/tabzr/Documents/CMU Dataset/r4.2/" + "r42_features_complex.h5" #+ name

print("Loading:", filename,flush=True)
joint = pd.read_hdf(filename, "table")
users = np.unique(joint.index.values)
print("There are", users.size, "users", flush=True)
num_features = np.unique(joint["feature"].values).size
print("Using", num_features, "features", flush=True)

assert(len(names) == len(saves))
for inert in inertias:

    userMatrices = {}

    # filename = "/home/tabz/Documents/CMU_Dataset/r4.2/" + "r42_features_complex.h5" #+ name
    # # filename = "C:/Users/tabzr/Documents/CMU Dataset/r4.2/" + "r42_features_complex.h5" #+ name

    # print("Loading:", filename,flush=True)
    # joint = pd.read_hdf(filename, "table")
    # users = np.unique(joint.index.values)
    # print("There are", users.size, "users", flush=True)
    # num_features = np.unique(joint["feature"].values).size
    # print("Using", num_features, "features", flush=True)

    # The initial configurations for the hidden markov model
    num_states = 10
    num_symbols = num_features
    deviation_from_uniform = 1/2
    # Seed the rng for reproducibility
    rng_seed = 42
    np.random.seed(rng_seed)

    def init_matrices():
        # Start probabilities
        startprobs = np.zeros(num_states)
        startprobs[0] = 1
        startprobs.fill(1/num_states)
        startprobs += np.random.rand(num_states) * deviation_from_uniform
        startprobs /= np.sum(startprobs)

        # Transition probabilities
        transprobs = np.empty((num_states,num_states))
        transprobs.fill(1/num_states)
        transprobs += np.random.rand(num_states,num_states) * deviation_from_uniform
        transprobs /= transprobs.sum(axis=1)[:,np.newaxis]

        # Emission probabilities
        emissionprobs = np.empty((num_states,num_symbols))
        emissionprobs.fill(1/num_symbols)
        emissionprobs += np.random.rand(num_states,num_symbols) * deviation_from_uniform
        emissionprobs /= emissionprobs.sum(axis=1)[:,np.newaxis]

        return startprobs, transprobs, emissionprobs

    def compute_probs(user_df):
        dayGrouping = pd.Grouper(key="date", freq="1D")
        weekGrouping = pd.Grouper(key="date", freq="1W")
        timeGrouping = user_df.groupby(weekGrouping)

        # print("Starting on user: ", user)
        s,t,e = init_matrices()
        model = HiddenMarkovModel.HMM(num_states, t,e,s)

        trainingPeriod = 4
        timesTrained = 0

        logProbScores = []
        matrices = []
        def mm(x,y,z):
            return (np.copy(x), np.copy(y), np.copy(z))
        matrices.append(mm(t,e,s))

        for name, group in timeGrouping:

            #The sequence for the time grouping we are considering
            seq = group["feature"].values

            if len(seq) < 1:
                # If there is no activity for this week
                logProbScores.append(0)
                continue

            if timesTrained > trainingPeriod:
                logProb = model.sequence_log_probability(seq)
                logProbScores.append(-logProb)

                #Train the model on the sequence we have just seen
            model.learn(seq, max_iters=20, threshold=0.01, restart_threshold=0.1,max_restarts=5, inertia=inert)
            matrices.append(mm(model.transitions, model.emissions, model.starts))
            timesTrained+=1

        return (logProbScores, matrices)

    userScores = {}

    def setInDict(u):
        def z(scoresAndMatrices):
            r = scoresAndMatrices[0]
            m = scoresAndMatrices[1]
            userScores[u] = r
            userMatrices[u] = m
        return z

    pool = mp.Pool(processes=8)
    print("Queueing up jobs", flush=True)
    for user in tqdm(users):
        pool.apply_async(compute_probs, args=(joint.loc[ joint.index == user ],), callback=setInDict(user))
    #     compute_probs(joint.loc[user])
    print("Progress on those jobs:", flush=True)
    done = 0
    for i in tqdm(users):
        while done >= len(userScores):
            time.sleep(1)
        done += 1
    pool.close()
    pool.join()
    pool.terminate()
    
    save = "userScores_simple_r42_" + str(inert) + "_inertia_v2.json"
    print("Saving:", save, flush=True)
    json.dump(userScores, open(save, "w"))

    print("Saving matrices", flush=True)
    np.save("r4.2_simple_matrices", userMatrices)
