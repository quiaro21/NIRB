using Flux, Random
function NN_mapping_full(from,to,epochs,perc_val,basis)
################# CHANGE INPUT ACCORDINGLY #############
    epochs = epochs
    #k = 25 # Number of reduced coefficients
    k=size(to)[1]
    input_size = size(from)[1] # Size of each input model
    hidden_layers = [200, 200] # Just an example, adjust based on your problem

    M=from
    A=to
    
    model = Chain(
    Dense(input_size, 2048, gelu),   # First hidden layer
    BatchNorm(2048),
    Dropout(0.20),
    Dense(2048, 512, gelu),         # Second hidden layer
    BatchNorm(512),
    Dense(512, k)          # Output layer with 100 units (linear activation)
)

#= BEST MODEL
    model = Chain(
    Dense(input_size, 2048, gelu),   # First hidden layer
    BatchNorm(2048),
    Dropout(0.30),
    Dense(2048, 512, gelu),         # Second hidden layer
    BatchNorm(512),
    Dense(512, k)          # Output layer with 100 units (linear activation)
)
    =#

    num_samples = size(M, 2) # Assuming M is 90x10000, this will be 10000
    num_train = floor(Int, num_samples * perc_val) # 90% of 10000 for training
    num_val = num_samples - num_train # Remaining for validation

    # Splitting M
    M_train = M[:, 1:10000]
    M_val = M[:, 10000+1:end]

    # Splitting A
    A_train = A[:, 1:10000]
    A_val = A[:, 10000+1:end]

    # With DataLoader for mini-batching
    batch_size = 256 # Adjust based on your preference and memory constraints
    train_dataloader = Flux.DataLoader((M_train, A_train), batchsize=batch_size, shuffle=true)
    val_dataloader = Flux.DataLoader((M_val, A_val), batchsize=batch_size) # Shuffle might not be necessary for validation

    function l2_reg(lambda, params)
        sum(norm(p)^2 for p in params) * lambda
    end

    function l1_reg(lambda, params)
        sum(norm(p) for p in params) * lambda
    end

    #penalty() = sum(abs2, model.W) + sum(abs2, model.b)

    lambda = 0.0001  # Regularization strength

    #loss(x, y) = Flux.huber_loss(basis*model(x), basis*y;delta=1,agg=mean) #+ l1_reg(lambda, Flux.params(model))
    loss(x, y) = Flux.mse(basis*model(x), basis*y) #+ l1_reg(lambda, Flux.params(model))
    
    optimizer = NAdam(0.002, (0.89, 0.995))
    decay_rate = 0.5
    decay_step = 2000
    # Learning rate is 0.001
    #optimizer = NAdam(0.001) # Learning rate is 0.001

    train_losses = Float64[]
    val_losses = Float64[]
    avg_train_loss=[]
    avg_val_loss=[]
    for epoch in 1:epochs
        if epoch % decay_step == 0 
            @info "Epoch $epoch: updating learning rate"
            optimizer.eta *= decay_rate  # Adjust the learning rate
        end
        # Training phase
        total_train_loss = 0.0
        for (x, y) in train_dataloader
            l = loss(x, y)
            total_train_loss += l
            Flux.train!(loss, Flux.params(model), [(x, y)], optimizer)
        end
        avg_train_loss = total_train_loss / length(train_dataloader)
        push!(train_losses, avg_train_loss)

        # Validation phase
        total_val_loss = 0.0
        for (x, y) in val_dataloader
            l = loss(x, y)
            total_val_loss += l
        end
        avg_val_loss = total_val_loss / length(val_dataloader)
        push!(val_losses, avg_val_loss)

        println("Epoch $epoch, Train Loss: $avg_train_loss, Val Loss: $avg_val_loss")
    end
    return model,train_losses,val_losses
end
function NN_mapping(from,to,epochs,perc_val)#,basis)
################# CHANGE INPUT ACCORDINGLY #############
    epochs = epochs
    #k = 25 # Number of reduced coefficients
    k=size(to)[1]
    input_size = 50 # Size of each input model
    hidden_layers = [200, 200] # Just an example, adjust based on your problem

    M=from
    A=to
    
    #model = Chain(
    #    Dense(input_size, hidden_layers[1], gelu),
    #    Dropout(0.20),  # 25% probability of dropping a unit
    #    Dense(hidden_layers[1], hidden_layers[2], gelu),
    #    Dropout(0.20),#15%
    #    Dense(hidden_layers[1], hidden_layers[2], gelu),
    #    #I have this model in flux
    #    Dropout(0.0),
    #    Dense(hidden_layers[2], k)
    #)
    model = Chain(
        Dense(50, 128,gelu),
        BatchNorm(128),
        Dropout(0.2),    
        Dense(128, 128,gelu),
        BatchNorm(128),
        Dropout(0.2),
        Dense(128, 64,gelu),
        Dense(64, k)
    )


    num_samples = size(M, 2) # Assuming M is 90x10000, this will be 10000
    num_train = floor(Int, num_samples * perc_val) # 90% of 10000 for training
    num_val = num_samples - num_train # Remaining for validation

    # Splitting M
    M_train = M[:, 1:10000]
    M_val = M[:, 10000+1:end]

    # Splitting A
    A_train = A[:, 1:10000]
    A_val = A[:, 10000+1:end]

    # With DataLoader for mini-batching
    batch_size = 128 # Adjust based on your preference and memory constraints
    train_dataloader = Flux.DataLoader((M_train, A_train), batchsize=batch_size, shuffle=true)
    val_dataloader = Flux.DataLoader((M_val, A_val), batchsize=batch_size) # Shuffle might not be necessary for validation

    function l2_reg(lambda, params)
        sum(norm(p)^2 for p in params) * lambda
    end
    lambda = 0.001  # Regularization strength

    loss(x, y) = Flux.mae(model(x), y) #+ l2_reg(lambda, Flux.params(model))
    optimizer = NAdam(0.002, (0.89, 0.995))
    decay_rate = 0.5
    decay_step = 2000
    # Learning rate is 0.001
    #optimizer = NAdam(0.001) # Learning rate is 0.001

    train_losses = Float64[]
    val_losses = Float64[]
    avg_train_loss=[]
    avg_val_loss=[]
    for epoch in 1:epochs
        if epoch % decay_step == 0 
            @info "Epoch $epoch: updating learning rate"
            optimizer.eta *= decay_rate  # Adjust the learning rate
        end
        # Training phase
        total_train_loss = 0.0
        for (x, y) in train_dataloader
            l = loss(x, y)
            total_train_loss += l
            Flux.train!(loss, Flux.params(model), [(x, y)], optimizer)
        end
        avg_train_loss = total_train_loss / length(train_dataloader)
        push!(train_losses, avg_train_loss)

        # Validation phase
        total_val_loss = 0.0
        for (x, y) in val_dataloader
            l = loss(x, y)
            total_val_loss += l
        end
        avg_val_loss = total_val_loss / length(val_dataloader)
        push!(val_losses, avg_val_loss)

        println("Epoch $epoch, Train Loss: $avg_train_loss, Val Loss: $avg_val_loss")
    end
    return model,train_losses,val_losses
end

function normalize_mat(mat)
    mean_mat=mean(mat)
    std_mat=std(mat)
    norm_mat = (mat .- mean_mat) ./ std_mat
    return norm_mat,mean_mat,std_mat
end

function denormalize_mat(norm_mat,mean,std)
    denorm_mat= (norm_mat .* std) .+ mean
    return denorm_mat
end

function app_pha_separation2D(S)
    resmat = zeros(2500,size(S,2))
    phamat = zeros(2500,size(S,2))
    for i in 1:size(S,2)
        a= reshape(S[:,i],(100,50))'
        tempres = a[1:25,:]
        temppha = a[26:end,:]
        resmat[:,i] = vec(tempres)
        phamat[:,i] = vec(temppha)
    end
    return log10.(resmat),phamat
end

function error_vec(mat_a,mat_b,idx1,idx2,mean,std)
    error_vec = zeros(idx2-idx1 +1) 
    for i in idx1:idx2
        a=reshape(mat_a[:,i],(25,100))
        b=denormalize_mat(reshape(mat_b[:,i],(25,100)),mean,std)
        #epsilon= error_cell(a,b)
        c = norm( a .- b)
        error_vec[i-10000]=c / norm(a)
    end
    return error_vec
end