# train plda model, save and load the model, enroll, generate scores

import bob.learn.em
import numpy
import pickle
import bob

data1 = numpy.array(
     [[3,-3,100],
      [4,-4,50],
      [40,-40,150]], dtype=numpy.float64)
data2 = numpy.array(
     [[3,6,-50],
      [4,8,-100],
      [1, 4, 62],
      [40,79,-800]], dtype=numpy.float64)
dummy_data = [data1,data2]

embedding_dim = 3
F_rank = 1
G_rank = 2

pldabase = bob.learn.em.PLDABase(embedding_dim, F_rank, G_rank)

trainer = bob.learn.em.PLDATrainer()

bob.learn.em.train(trainer, pldabase, dummy_data, max_iterations=40)

# saver
f = bob.io.base.HDF5File('plda_train.hdf5', 'w')
# save model
pldabase.save(f)

print(pldabase)

# main plda trained model
plda = bob.learn.em.PLDAMachine(pldabase)



# loader
f = bob.io.base.HDF5File('plda_train.hdf5', 'r')
pldabase_2 = bob.learn.em.PLDABase(embedding_dim, F_rank, G_rank)
pldabase_2.load(f)

# main plda trained model loaded
plda_2 = bob.learn.em.PLDAMachine(pldabase_2)

# main plda trained model loaded
plda_2 = bob.learn.em.PLDAMachine(pldabase_2)

# dummy no weight
pldabase_3 = bob.learn.em.PLDABase(embedding_dim, F_rank, G_rank)
plda_3 = bob.learn.em.PLDAMachine(pldabase_3)

# check plda results for online and loaded model
samples = numpy.array(
     [[3.5,-3.4,102],
      [4.5,-4.3,56]], dtype=numpy.float64)

loglike = plda.compute_log_likelihood(samples)

loglike_2 = plda_2.compute_log_likelihood(samples)

loglike_3 = plda_3.compute_log_likelihood(samples)

print(loglike)
print('---------------')
print(loglike_2)
print('---------------')
print(loglike_3)
