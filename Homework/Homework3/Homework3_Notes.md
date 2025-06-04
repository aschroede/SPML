# Homework III Summary

Code Instructions: 
- Implement backdoor attack in FL using Blend trigger
- Code Distributed Backdoor Attack (DB)
- Implement a Scaling Attack
- Compare these attacks
- For all attacks us CIFAR-10
- Do not need to train clean mode, make use of pre-trained ResNet18Light from tutorial.

Report Instructions
- Make sure to describe what is done, why it is done that way, what the results are, and what they mean. 
- Thorough discussion of results - not only waht is observed by why. 
- Include tables/figures to present the results
- Make sure code runs out of the box. 
- Mention source of your code even if code is your own or taken from tutorial. 
- Copy sub-questions you are answering into the notebook for more clarity.

## Federated Learning Summary
Federated learning is a type of machine learning where multiple decices train a shared model without sharing their raw data with each other. **This allows training across decentralized data sources like smartphones or hospitals while preservering privacy, reducing latency, and minimizing bandwidth usage**

### How it works
1. **Initialization**: a central server creates the base ML model and sends it to participating devices
2. **Local Training:**: each device train the model using its own local data
3. **Model updates:** the devices send the updated model parameters (gradients/weights) - not the raw data - back to the server
4. **Aggregation:** the server aggregates all the updates (often using FedAvg) to produce new gloval model
5. **Iteration**: Steps 2-4 are repeated until the model converges.


## Background From Tutorial 8
1. Each client has a main label and 90% of the data the client is trained on comes from that label while 10% is randomly sampled. This rate is defined by `IID_RATE`
2. We use a pre-trained `ResNet-18` to avoid long training times. 
3. Note that you cannot simply compare or combine tensors stored on different devices (CPU or GPU). Please keep this in mind.
4. Also, be aware of the difference between a model object and a state-dict. The state-dict is just a dictionary containing the layers and their trainable parameters. To do a prediction, you need to load the state-dict into a model object (instances of Resnet18Light).


## Fixing the Copy Bug

Ruben purposes to replace the code in this code block with the following:

for client_index in range(NUMBER_OF_BENIGN_CLIENTS):
    print_timed(f'Client {client_index}')
    trained_weights = client_training(global_model_state_dict, local_model, ...)
    all_trained_benign_weights.append(copy.deepcopy(trained_weights))
Where you add copy.deepcopy(), around the trained_weights variable. This ensures you actually append a copy of the object instead of a reference. This should also work when using GPU. 

