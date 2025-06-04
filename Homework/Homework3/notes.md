# Bug Fix

Hi all,

Ruben te Morsche figured out what caused the difference in accuracy for the Federated Learning tutorial notebook when using CPU vs GPU.

In the notebook, we go over the number of clients, perform local training and then append the "trained_weights" to a list named "all_trained_benign_weights". The code cell just below "Federated Learning (Simulation)".

There is a difference what is being appended to the list when using CPU or GPU.

With GPU, in the background, implicitly, a new object is created and so the list indeed contains the results of each client training. But with CPU, no new object is created and a reference is stored. When you then perform additional training, all references are updated. 

Ruben purposes to replace the code in this code block with the following:

for client_index in range(NUMBER_OF_BENIGN_CLIENTS):
    print_timed(f'Client {client_index}')
    trained_weights = client_training(global_model_state_dict, local_model, ...)
    all_trained_benign_weights.append(copy.deepcopy(trained_weights))
Where you add copy.deepcopy(), around the trained_weights variable. This ensures you actually append a copy of the object instead of a reference. This should also work when using GPU. 

Using this solution should indeed result in an accuracy of around 80% if your aggregation code is correct.

Thank you Ruben for finding and fixing this bug!