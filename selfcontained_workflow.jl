using CSV, DataFrames, Statistics, LinearAlgebra, Plots
include("DRIVERS_NN.jl")

#Fixing the random seed to ensure reproducibility
Random.seed!(1234)
#Functions: Ignore, you might not need to modify this.
epochs = 1
rank= 100
#START READING HERE:

#Here we are loading the surrogate models from the disk
println("1. Loading the models ")
#M_surr= Matrix(CSV.read("M_surr.csv", DataFrame))
M= Matrix(CSV.read("M_krig_julia.csv", DataFrame))
#Here we are loading the snapshot matrices from the disk and splitting into 
#resistivity and phase matrices for TE and TM modes.
println("2. Loading the snapshot matrices ")
S_tm = Matrix(CSV.read("snapshot_matrixTM.csv", DataFrame,header=false))
S_te = Matrix(CSV.read("snapshot_matrixTE.csv", DataFrame,header=false))

resmat_tm,phamat_tm = app_pha_separation2D(S_tm)
resmat_te,phamat_te = app_pha_separation2D(S_te)
phamat_te=phamat_te .-180

#Normalization of the snapshot matrices, and storing the de-normalization parameters:
println("3. Normalizing the snapshot matrices and storing de-normalization parameters")
#TM mode
resmat_norm_tm,meanres_tm,stdres_tm = normalize_mat(resmat_tm)
phamat_norm_tm,meanpha_tm,stdpha_tm = normalize_mat(phamat_tm)
Snorm_tm = vcat(resmat_norm_tm,phamat_norm_tm)

#TE mode
resmat_norm_te,meanres_te,stdres_te = normalize_mat(resmat_te)
phamat_norm_te,meanpha_te,stdpha_te = normalize_mat(phamat_te)
Snorm_te = vcat(resmat_norm_te,phamat_norm_te)

#Basis creation
println("4. Building the base using the SVD ")
#TM mode
Us_n_res_tm,SVs_n_res_tm,Vs_n_res_tm = svd(Snorm_tm[1:2500,1:10000])
Us_n_pha_tm,SVs_n_pha_tm,Vs_n_pha_tm = svd(Snorm_tm[2501:5000,1:10000])
#TE mode
Us_n_res_te,SVs_n_res_te,Vs_n_res_te = svd(Snorm_te[1:2500,1:10000])
Us_n_pha_te,SVs_n_pha_te,Vs_n_pha_te = svd(Snorm_te[2501:5000,1:10000])

#Calculate the reduced coefficients:
println("5. Calculate the reduced coefficients by projecting the snapshot matrix onto the basis")
#TM mode
Anorm_res_tm=Us_n_res_tm'*Snorm_tm[1:2500,:]
Anorm_pha_tm=Us_n_pha_tm'*Snorm_tm[2501:end,:]

#TE mode
Anorm_res_te=Us_n_res_te'*Snorm_te[1:2500,:]
Anorm_pha_te=Us_n_pha_te'*Snorm_te[2501:end,:]

#Here we are training the neural networks:

println("6. Training the neural networks with $epochs epochs and $rank reduced coefficients")
#For the TM mode
gsn_res_tm,train_loss_sn_res_tm,val_loss_sn_res_tm=NN_mapping_full(M,Anorm_res_tm[1:rank,:],epochs,0.98,Us_n_res_tm[:,1:nbasis])
gsn_pha_tm,train_loss_sn_pha_tm,val_loss_sn_pha_tm=NN_mapping_full(M,Anorm_pha_tm[1:rank,:],epochs,0.98,Us_n_pha_tm[:,1:nbasis])
#For the TE mode
gsn_res_te,train_loss_sn_res_te,val_loss_sn_res_te=NN_mapping_full(M,Anorm_res_te[1:rank,:],epochs,0.98,Us_n_res_te[:,1:nbasis])
gsn_pha_te,train_loss_sn_pha_te,val_loss_sn_pha_te=NN_mapping_full(M,Anorm_pha_te[1:rank,:],epochs,0.98,Us_n_pha_te[:,1:nbasis])

println("7. Run forward modeling using the newly trained network")

S_rb_k_norm_res_tm = Us_n_res_tm[:,1:rank]*gsn_res_tm(M)[1:rank,:]
S_rb_k_norm_pha_tm = Us_n_pha_tm[:,1:rank]*gsn_pha_tm(M)[1:rank,:]
S_rb_k_norm_res_te = Us_n_res_te[:,1:rank]*gsn_res_te(M)[1:rank,:]
S_rb_k_norm_pha_te = Us_n_pha_te[:,1:rank]*gsn_pha_te(M)[1:rank,:]

idx=10001

println("8. De-normalize the snapshot matrices obtained with the Neural Networks, for model $idx")
println("NOTE: Validation models are from idx 10001 to idx 10200")

HiFi_res_tm=reshape(resmat_tm[:,idx],(25,100))
HiFi_res_te=reshape(resmat_te[:,idx],(25,100))
NN_res_tm=denormalize_mat(reshape(S_rb_k_norm_res_tm[:,idx],(25,100)),meanres_tm,stdres_tm)
NN_res_te=denormalize_mat(reshape(S_rb_k_norm_res_te[:,idx],(25,100)),meanres_te,stdres_te)

HiFi_pha_tm=reshape(phamat_tm[:,idx],(25,100))
HiFi_pha_te=reshape(phamat_te[:,idx],(25,100))
NN_pha_tm=denormalize_mat(reshape(S_rb_k_norm_pha_tm[:,idx],(25,100)),meanpha_te,stdpha_te)
NN_pha_te=denormalize_mat(reshape(S_rb_k_norm_pha_te[:,idx],(25,100)),meanpha_te,stdpha_te)

println("9. Error calculation for the 200 validation models")

median_err_res_tm = error_vec(resmat_tm,S_rb_k_norm_res_tm,10001,10200,meanres_tm,stdres_tm)
median_err_pha_tm = error_vec(phamat_tm,S_rb_k_norm_pha_tm,10001,10200,meanpha_tm,stdpha_tm)
median_err_res_te = error_vec(resmat_te,S_rb_k_norm_res_te,10001,10200,meanres_te,stdres_te)
median_err_pha_te = error_vec(phamat_te,S_rb_k_norm_pha_te,10001,10200,meanpha_te,stdpha_te)

println("10. Plotting the results")

title= plot(title = "Error Histograms: $epochs epochs, rank $rank", grid = false, showaxis = false, bottom_margin = -30Plots.px)
plt1=histogram(median_err_res_tm,title="Res TM",bins=20)
plt2=histogram(median_err_pha_tm,title="Pha TM",bins=20)
plt3=histogram(median_err_res_te,title="Res TE",bins=20)
plt4=histogram(median_err_pha_te,title="Pha TE",bins=20)
plot(title,plt1,plt2,plt3,plt4,layout=@layout([A{0.01h};[A B;C D]]),right_margin=20Plots.px)
