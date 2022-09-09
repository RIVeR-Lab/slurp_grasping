import torch
from torch import nn


class ContNet(nn.Module):

    def __init__(self, num_features, num_containers, num_contents, container_hidden_layers, contents_hidden_layers, use_dropout=False):
        super(ContNet, self).__init__()
        # Construct the activation function
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        # Drop is used to assist with generalization
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.10)

        # Create the Container network
        self.container_input = nn.Linear(num_features, container_hidden_layers[0])
        # Create the the container feed-forward network
        self.container_list = nn.ModuleList([
            nn.Linear(container_hidden_layers[i-1], container_hidden_layers[i]) 
            for i in range(1, len(container_hidden_layers))
        ])
        self.container_output = nn.Linear(container_hidden_layers[-1], num_containers)

        # Create the Content Network
        # Input here takes the original spectral signature and the output of the previous network
        self.content_input = nn.Linear(num_features+num_containers, contents_hidden_layers[0])
        # Create the the container feed-forward network
        self.content_list = nn.ModuleList([
            nn.Linear(contents_hidden_layers[i-1], contents_hidden_layers[i]) 
            for i in range(1, len(contents_hidden_layers))
        ])
        self.content_output = nn.Linear(contents_hidden_layers[-1], num_contents)
        
    def forward(self, spec_data):

        # Pass the spectrometer data through the network
        x = self.container_input(spec_data)
        # Apply dropout here
        if self.use_dropout:
            x = self.dropout(x)
        # Pass through activation function
        x = self.relu(x)

        # ========= Container Network =========
        for layer in self.container_list:
            x = layer(x)
            # Keep applying the dropout!
            if self.use_dropout:
                x = self.dropout(x)
            x = self.relu(x)
            
        # Calculate the container probs - this will also be optimized
        container_probs = self.container_output(x)
        # ========= Contents Network =========
        x = self.content_input(torch.cat((spec_data,container_probs),1))
        # Apply dropout here
        if self.use_dropout:
            x = self.dropout(x)
        # Pass through activation function
        x = self.relu(x)
        
        for layer in self.content_list:
            x = layer(x)
            # Keep applying the dropout!
            if self.use_dropout:
                x = self.dropout(x)
            x = self.relu(x)
            
        # Calculate the container probs - this will also be optimized
        content_probs = self.content_output(x)

        return [container_probs, content_probs]