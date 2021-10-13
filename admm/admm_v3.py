import numpy as np
import torch
from numpy import linalg as LA

import wandb

# --------------------------------------------------------------
def apply_quantization(args, model, device=None, name_list=None):
    for name, weight in model.named_parameters():
        if name in name_list:
            quantized_weight = weight.cpu().detach().numpy()
            Q = model.Q[name].cpu().detach()
            
            if args.quant_type == "binary":
                quantized_weight[quantized_weight > 0] = model.alpha[name]
                quantized_weight[quantized_weight <= 0] = -model.alpha[name]
                weight.data = torch.Tensor(quantized_weight).to(device)
            
            elif args.quant_type == "ternary":
                quantized_weight = np.where(quantized_weight > 0.5 * model.alpha[name], model.alpha[name],
                                            quantized_weight)
                quantized_weight = np.where(quantized_weight < -0.5 * model.alpha[name], -model.alpha[name],
                                            quantized_weight)
                quantized_weight = np.where((quantized_weight <= 0.5 * model.alpha[name]) & (quantized_weight >= -0.5 * model.alpha[name]), 
                                                0, quantized_weight)

                # quantized_weight = np.where(quantized_weight > 0.3 * model.alpha[name], model.alpha[name],
                #                             quantized_weight)
                # quantized_weight = np.where(quantized_weight < -0.3 * model.alpha[name], -model.alpha[name],
                #                             quantized_weight)
                # quantized_weight = np.where(
                #     (quantized_weight <= 0.3 * model.alpha[name]) & (quantized_weight >= -0.3 * model.alpha[name]), 0,
                #     quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "fixed":
                quantized_weight = model.alpha[name] * Q
                half_num_bits = model.num_bits[name] - 1
                centroids = []
                for value in range(-2 ** half_num_bits - 1, 2 ** half_num_bits):
                    centroids.append(value)

                for i, value in enumerate(centroids):
                    if i == 0:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] < value + 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) - 1:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] >= value - 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    else:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= value - 0.5) & (
                                quantized_weight / model.alpha[name] < value + 0.5), value * model.alpha[name],
                                                    quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)

            elif args.quant_type == "one-side":
                num_bits = args.num_bits  # include zero, one side
                centroids = []
                for value in range(-2 ** num_bits, 2 ** num_bits + 1):
                    if value == 0: continue
                    centroids.append(value)
                for i, value in enumerate(centroids):
                    if i == 0:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] < value + 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) - 1:
                        quantized_weight = np.where(quantized_weight / model.alpha[name] >= value - 0.5,
                                                    value * model.alpha[name], quantized_weight)
                    elif i == len(centroids) / 2 - 1:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= value - 0.5) & (
                                quantized_weight < 0), value * model.alpha[name],
                                                    quantized_weight)
                    elif i == len(centroids) / 2:
                        quantized_weight = np.where((quantized_weight > 0) & (
                                quantized_weight / model.alpha[name] < value + 0.5), value * model.alpha[name],
                                                    quantized_weight)
                    else:
                        quantized_weight = np.where((quantized_weight / model.alpha[name] >= value - 0.5) & (
                                quantized_weight / model.alpha[name] < value + 0.5), value * model.alpha[name],
                                                    quantized_weight)
                weight.data = torch.Tensor(quantized_weight).to(device)
            else:
                raise ValueError("The quantized type is not supported!")


# --------------------------------------------------------------
def project_to_centroid(args, model, W, name, device):
    U = model.U[name].cpu().detach().numpy()
    Q = model.Q[name].cpu().detach().numpy()

    # alpha = np.mean(np.abs(W))  # initialize alpha
    alpha = model.alpha[name]
    num_iter_quant = 5  # default 20, 5 in paper

    if args.quant_type == "binary":
        Q = np.where((W + U) > 0, 1, -1)
        alpha = np.sum(np.multiply((W + U), Q))
        QtQ = np.sum(Q ** 2)
        alpha /= QtQ

    elif args.quant_type == "ternary": #2-bit
        for n in range(num_iter_quant):

            Q = np.where((W + U) / alpha > 0.5, 1, Q)
            Q = np.where((W + U) / alpha < -0.5, -1, Q)
            Q = np.where(((W + U) / alpha >= -0.5) & ((W + U) / alpha <= 0.5), 0, Q)

            # Q = np.where((W + U) / alpha > 0.3, 1, Q)
            # Q = np.where((W + U) / alpha < -0.3, -1, Q)
            # Q = np.where(((W + U) / alpha >= -0.3) & ((W + U) / alpha <= 0.3), 0, Q)

            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.sum(Q ** 2)
            alpha /= QtQ

    elif args.quant_type == "fixed": #try 4 or 1
        half_num_bits = model.num_bits[name] - 1
        centroids = []
        for value in range(-2 ** half_num_bits + 1, 2 ** half_num_bits):
            centroids.append(value)

        for n in range(num_iter_quant):
            Q = np.where(np.round((W + U) / alpha) <= centroids[0], centroids[0], Q)
            Q = np.where(np.round((W + U) / alpha) >= centroids[-1], centroids[-1], Q)
            Q = np.where((np.round((W + U) / alpha) < centroids[-1]) & (np.round((W + U) / alpha) > centroids[0]),
                         np.round((W + U) / alpha), Q)

            # for i, value in enumerate(centroids):
            #
            #     if i == 0:
            #         Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
            #     elif i == len(centroids) - 1:
            #         Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
            #     else:
            #         Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value,
            #                      Q)

            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.multiply(Q, Q)
            QtQ = np.sum(QtQ)
            alpha /= QtQ

    elif args.quant_type == "one-side":
        num_bits = args.num_bits  # not include zero, one side
        alpha = alpha / ((1 + 2 ** num_bits) * (2 ** (num_bits - 1)))
        centroids = []
        for value in range(-2 ** num_bits, 2 ** num_bits + 1):
            if value == 0: continue
            centroids.append(value)

        for n in range(num_iter_quant):
            for i, value in enumerate(centroids):
                if i == 0:
                    Q = np.where(((W + U) / alpha) < (value + 0.5), value, Q)
                elif i == len(centroids) - 1:
                    Q = np.where(((W + U) / alpha) >= (value - 0.5), value, Q)
                elif i == len(centroids) / 2 - 1:
                    Q = np.where((((W + U) / alpha) >= (value - 0.5)) & ((W + U) < 0), value, Q)
                elif i == len(centroids) / 2:
                    Q = np.where(((W + U) >= 0) & ((W + U) / alpha < (value + 0.5)), value, Q)
                else:
                    Q = np.where((((W + U) / alpha) >= (value - 0.5)) & (((W + U) / alpha) < (value + 0.5)), value, Q)
            alpha = np.sum(np.multiply((W + U), Q))
            QtQ = np.sum(Q ** 2)
            alpha /= QtQ

    model.U[name] = torch.Tensor(U).to(device)
    model.Q[name] = torch.Tensor(Q).to(device)
    model.Z[name] = alpha * model.Q[name]
    model.alpha[name] = alpha

    return model.Z[name], model.alpha[name], model.Q[name]


# --------------------------------------------------------------
def admm_initialization(args, model, device, name_list=None):
    init_rho = args.init_rho

    model.alpha = {}
    model.rhos = {}
    model.U = {}
    model.Q = {}  # alpha * Q = Z
    model.Z = {}
    model.ce = 0 #cross entropy loss
    model.ce_prev = 0 #previous cross ent. loss


    if args.optimization =='savlr':

        model.W_prev = {} # previous W that satisfied surrogate opt. condition
        model.Z_prev = {} # previous Z that satisfied surrogate opt. condition
        
        model.ADMM_Lambda = {} # SLR multiplier
        model.s = args.initial_s # stepsize
        model.alpha_slr = 0
        model.k = 1 #SLR 

        model.condition1 = []
        model.condition2 = []
      

    for name, param in model.named_parameters():
        if "weight" in name and name in name_list:
            print(name)
            model.rhos[name] = init_rho
            weight = param.cpu().detach().numpy()
            if args.quant_type == "binary" or args.quant_type == "ternary":
                model.alpha[name] = np.mean(np.abs(weight))  # initialize alpha
                # model.alpha[name] = np.mean(np.abs(weight[np.nonzero(weight)]))  # initialize alpha
            elif args.quant_type == "fixed":
                model.alpha[name] = np.mean(np.abs(weight[np.nonzero(weight)]))
                half_num_bits = model.num_bits[name] - 1
                model.alpha[name] = model.alpha[name] / ((2 ** half_num_bits - 1) / 2)
            model.U[name] = torch.zeros(param.shape).to(device)
            model.Q[name] = torch.zeros(param.shape).to(device)
            updated_Z, updated_aplha, updated_Q = project_to_centroid(args, model, weight, name, device)
            model.Z[name] = updated_Z
            
            if args.optimization =='savlr':
                model.W_prev[name] =  torch.zeros(param.shape).cuda()
                model.Z_prev[name] =  torch.zeros(param.shape).cuda()
                model.ADMM_Lambda[name] = torch.zeros(param.shape).cuda()  


# --------------------------------------------------------------
def z_u_update(args, model, device, epoch, iteration, batch_idx, name_list=None):
    
    # print("!!!!!! Optimization Method: ", args.optimization)

    if args.optimization == 'admm':
        if (epoch != 1) and ((epoch - 1) % args.admm_epochs == 0) and (batch_idx == 0):
            print("Updating Z, U!!!!!!")
            for name, W in model.named_parameters():
                if ("weight" not in name) or (name not in name_list):
                    continue
                Z_prev = torch.clone(model.Z[name])
                weight = W.cpu().detach().numpy()
                updated_Z, _, _ = project_to_centroid(args, model, weight, name,
                                                      device)  # equivalent to Euclidean Projection

                if args.update_rho == 1:
                    # print("++++++++++++ update rho")
                    model.rhos[name] = model.rhos[name] * 1.1
                elif args.update_rho == 0:
                    # print("++++++++++++ NOT update rho")
                    model.rhos[name] = model.rhos[name]

                if (args.verbose):
                    print("at layer {}. W(k+1)-Z(k+1): {}".format(name, torch.sqrt(
                        torch.sum((W - model.Z[name]) ** 2)).item()))
                    print("at layer {}, Z(k+1)-Z(k): {}".format(name, torch.sqrt(
                        torch.sum((model.Z[name] - Z_prev) ** 2)).item()))
            
                model.U[name] = W - model.Z[name] + model.U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)
                

    if args.optimization == 'savlr':

        if (epoch != 1) and (batch_idx%args.update_btch == 0):

            # save prev. values
            alpha_slr_prev = model.alpha_slr
            s_prev = model.s
            
            ADMM_Lambda_prev = {}
            total_n1 = 0  # hold for ||W(k-1)-Z(k-1)||
            total_n2 = 0  # hold for ||W(k)-Z(k-1)||
            for name, W in model.named_parameters():
                if ("weight" not in name) or (name not in name_list):
                    continue
                
                ### save prev. lambda
                ADMM_Lambda_prev[name] = model.ADMM_Lambda[name] 

                ### hold for s(k)
                n1 = torch.sqrt(torch.sum((model.W_prev[name]-model.Z_prev[name])**2)).item() #||W_n(k-1)-Z_n(k-1)||
                n2 = torch.sqrt(torch.sum((W-model.Z_prev[name])**2)).item()  #||W_n(k)-Z_n(k-1)||
                total_n1 += n1
                total_n2 += n2

            ##############
            ### step 1 ###
            ##############
            ### Eq2: update W using SGD  (can directly get W from model.name_parameters())

            ### Eq3: Check if surr. opt. condition is satisfied or k==1
            satisfied1 = Lagrangian1(model, name_list)
            model.condition1.append(satisfied1)
            wandb.log({"condition/Condition1": satisfied1})

            if (satisfied1 == 1) or (model.k == 1): 

                print("k = " + str(model.k))
                wandb.log({"Hyper/k": model.k})

                ### Eq6: alpha(k)
                pow = 1 - (1/(model.k**args.r))
                model.alpha_slr = 1 - (1/(args.M*(model.k**pow))) 
                print("savlr alpha:", model.alpha_slr)
                wandb.log({"Hyper/alpha_slr": model.alpha_slr})

                ### Eq5: s(k)
                if (total_n1 != 0) and (total_n2 != 0):  #if norms are not 0, update stepsize
                    model.s = model.alpha_slr * (model.s*total_n1/total_n2)   # Eq5: s(k)
                    print("savlr s:", model.s, "\n")
                    wandb.log({"Hyper/savlr_s": model.s})

                ### Eq4: lambda(k)
                for name, W in model.named_parameters():
                    if ("weight" not in name) or (name not in name_list):
                        continue                
                    model.ADMM_Lambda[name] = model.s*(W-model.Z_prev[name])+model.ADMM_Lambda[name]
                    model.U[name] = model.ADMM_Lambda[name]/model.rhos[name] 

                    # model.W_prev[name] = W     
                    # model.Z_prev[name] = model.Z[name]
                
                ### increase k
                model.k += 1 

            else:
                ### back to update W again
                return
            
            ##############
            ### step 2 ###
            ##############

            total_n1 = 0  # hold for ||W(k-1)-Z(k-1)||
            total_n2 = 0  # hold for ||W(k)-Z(k)||

            ### Eq7: update Z using projection
            for name, W in model.named_parameters():
                if ("weight" not in name) or (name not in name_list):
                    continue
                
                weight = W.cpu().detach().numpy()

                ## Eq8: equivalent to Euclidean Projection
                updated_Z, _, _ = project_to_centroid(args, model, weight, name,
                                                      device)

                if args.update_rho == 1:
                    model.rhos[name] = model.rhos[name] * 1.1
                    # print("++++++++++++ update rho")
                elif args.update_rho == 0:
                    model.rhos[name] = model.rhos[name]
                    # print("++++++++++++ NOT update rho")

                model.Z[name] = updated_Z   # Z(k)

                ### hold for s(k+1)
                n1 = torch.sqrt(torch.sum((model.W_prev[name]-model.Z_prev[name])**2)).item() #||W_n(k-1)-Z_n(k-1)||
                n2 = torch.sqrt(torch.sum((W-model.Z[name])**2)).item()  #||W_n(k)-Z_n(k)||
                total_n1 += n1
                total_n2 += n2


            ### Eq9: check if surr. opt. condition is satisfied or k==1
            satisfied2 = Lagrangian2(model, name_list)
            model.condition2.append(satisfied2)
            wandb.log({"condition/Condition2": satisfied2})

            if (satisfied2 == 1) or (model.k == 1): 

                print("k = " + str(model.k))
                wandb.log({"Hyper/k": model.k})

                ### Eq12: alpha(k+1)
                pow = 1 - (1/(model.k**args.r))
                model.alpha_slr = 1 - (1/(args.M*(model.k**pow)))
                print("savlr alpha:", model.alpha_slr)
                wandb.log({"Hyper/alpha_slr": model.alpha_slr })

                ### Eq11: s(k+1)
                if (total_n1 != 0) and (total_n2 != 0):  #if norms are not 0, update stepsize
                    model.s = model.alpha_slr * (model.s*total_n1/total_n2)   # Eq10: s(k)
                    print("savlr s:", model.s, "\n")
                    wandb.log({"Hyper/savlr_s": model.s})

                ### Eq10: lambda(k+1)
                for i, (name, W) in enumerate(model.named_parameters()):
                    if ("weight" not in name) or (name not in name_list):
                        continue
                    
                    model.ADMM_Lambda[name] = model.s*(W-model.Z[name]) + model.ADMM_Lambda[name]
               
                    model.W_prev[name] = W
                    model.Z_prev[name] = model.Z[name]
                    model.U[name] = model.ADMM_Lambda[name]/model.rhos[name] 

                ### increase k
                model.k += 1 

            else:
                ### drop all saved values (alpha, s, lambda, k) and restore to previous values
                model.k -= 1
                model.alpha_slr = alpha_slr_prev
                model.s = s_prev

                for name, W in model.named_parameters():
                    if ("weight" not in name) or (name not in name_list):
                        continue
                    model.ADMM_Lambda[name] = ADMM_Lambda_prev[name] 
                
                ### back to step 1 to update W (current W is not good enough for Z, so need new W)
                return
            
            iteration[0] = model.k

# --------------------------------------------------------------

# def Lagrangian1(model, name_list):
#     '''
#     This functions checks Surrogate optimality condition after each epoch. 
#     If the condition is satisfied, it returns 1.
#     If not, it returns 0.

#     '''

#     admm_loss = {}
#     admm_loss2 = {}
#     U_sum = 0 
#     U_sum2 = 0
#     satisfied = 0   # flag for satisfied condition

#     for i, (name, W) in enumerate(model.named_parameters()):

#         if ("weight" not in name) or (name not in name_list):
#             continue

#         U = model.ADMM_Lambda[name] / model.rhos[name]

#         admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W-model.Z_prev[name]+U, p=2)**2) #calculate current Lagrangian
#         U_sum = U_sum + (0.5 * model.rhos[name] * (torch.norm(U, p=2)**2))

#         admm_loss2[name] = 0.5 * model.rhos[name] * (torch.norm(model.W_prev[name]-model.Z_prev[name]+U, p=2)**2) #calculate prev Lagrangian
#         U_sum2 = U_sum2 + (0.5 * model.rhos[name] * (torch.norm(U, p=2)**2))

#     Lag = U_sum #current
#     Lag2 = U_sum2 #prev

#     Lag = model.ce + U_sum #current
#     Lag2 = model.ce_prev + U_sum2 #prev

#     #print("ce", model.ce)
#     #print("ce prev", model.ce_prev)
#     for k, v in admm_loss.items():
#         Lag += v 
        
#     for k, v in admm_loss2.items():
#         Lag2 += v 

    
#     if Lag < Lag2: #if current Lag < previous Lag
#         satisfied = 1
#         print("\ncondition1 satisfied\n")
#     else:
#         satisfied = 0
#         print("\ncondition1 not satisfied\n")

#     return satisfied


def Lagrangian1(model, name_list):
    '''
    This functions checks Surrogate optimality condition after each epoch. 
    If the condition is satisfied, it returns 1.
    If not, it returns 0.

    '''

    admm_loss = {}
    admm_loss2 = {}
    U_sum = {}
    satisfied = 0   # flag for satisfied condition

    for i, (name, W) in enumerate(model.named_parameters()):

        if ("weight" not in name) or (name not in name_list):
            continue

        admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W - model.Z_prev[name] + model.U[name], p=2)**2) #calculate current Lagrangian
        admm_loss2[name] = 0.5 * model.rhos[name] * (torch.norm(model.W_prev[name] - model.Z_prev[name] + model.U[name], p=2)**2) #calculate prev Lagrangian
        
        U_sum[name] = (0.5 * model.rhos[name] * (torch.norm(model.U[name], p=2)**2))


    Lag = model.ce                   # current
    Lag2 = model.ce_prev             # prev


    for k in admm_loss:
        Lag += (admm_loss[k]-U_sum[k])
        
    for k in admm_loss2:
        Lag2 += (admm_loss2[k]-U_sum[k])
    

    if Lag < Lag2: #if current Lag < previous Lag
        satisfied = 1
        print("\ncondition1 satisfied\n")
    else:
        satisfied = 0
        print("\ncondition1 not satisfied\n")

    return satisfied


# --------------------------------------------------------------
             
# def Lagrangian2(model, name_list):
#     '''
#     This functions checks Surrogate optimality condition after each epoch. 
#     If the condition is satisfied, it returns 1.
#     If not, it returns 0.

#     '''

#     admm_loss = {}
#     admm_loss2 = {}
#     U_sum = 0 
#     U_sum2 = 0
#     satisfied = 0 #flag for satisfied condition

#     for i, (name, W) in enumerate(model.named_parameters()):

#         if ("weight" not in name) or (name not in name_list):
#             continue

#         U = model.ADMM_Lambda[name] / model.rhos[name]

#         admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W-model.Z[name]+U, p=2) ** 2) #calculate current Lagrangian
#         U_sum = U_sum + (0.5 * model.rhos[name] * (torch.norm(U, p=2) ** 2))

#         admm_loss2[name] = 0.5 * model.rhos[name] * (torch.norm(W-model.Z_prev[name]+U, p=2) ** 2) #calculate prev Lagrangian
#         U_sum2 = U_sum2 + (0.5 * model.rhos[name] * (torch.norm(U, p=2) ** 2))

#     Lag = U_sum #current
#     Lag2 = U_sum2 #prev

#     Lag = model.ce + U_sum #current
#     Lag2 = model.ce_prev + U_sum2 #prev

#     #print("ce", model.ce)
#     #print("ce prev", model.ce_prev)
#     for k, v in admm_loss.items():
#         Lag += v 
        
#     for k, v in admm_loss2.items():
#         Lag2 += v 

 
#     if Lag < Lag2: #if current Lag < previous Lag
#         satisfied = 1
#         print("\ncondition2 satisfied\n")
#     else:
#         satisfied = 0
#         print("\ncondition2 not satisfied\n")

#     return satisfied


def Lagrangian2(model, name_list):
    '''
    This functions checks Surrogate optimality condition after each epoch. 
    If the condition is satisfied, it returns 1.
    If not, it returns 0.

    '''

    admm_loss = {}
    admm_loss2 = {}
    U_sum2 = {}
    satisfied = 0 #flag for satisfied condition

    for i, (name, W) in enumerate(model.named_parameters()):

        if ("weight" not in name) or (name not in name_list):
            continue

        admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W - model.Z[name] + model.U[name], p=2) ** 2) #calculate current Lagrangian
        admm_loss2[name] = 0.5 * model.rhos[name] * (torch.norm(W - model.Z_prev[name] + model.U[name], p=2) ** 2) #calculate prev Lagrangian
        
        U_sum2[name] = (0.5 * model.rhos[name] * (torch.norm(model.U[name], p=2) ** 2))


    Lag = model.ce #current
    Lag2 = model.ce_prev #prev


    for k in admm_loss:
        Lag += (admm_loss[k]-U_sum2[k])
        
    for k in admm_loss2:
        Lag2 += (admm_loss2[k]-U_sum2[k])


    if Lag < Lag2: #if current Lag < previous Lag
        satisfied = 1
        print("\ncondition2 satisfied\n")
    else:
        satisfied = 0
        print("\ncondition2 not satisfied\n")

    return satisfied


    
# --------------------------------------------------------------
def append_admm_loss(args, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name not in model.rhos:
            continue
        admm_loss[name] = 0.5 * model.rhos[name] * (torch.norm(W - model.Z[name] + model.U[name], p=2) ** 2)

    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss

# --------------------------------------------------------------
def test_sparsity(model):
    """
    test sparsity for every involved layer and the overall compression rate
    """

    layer = 0
    tot_param = 0
    tot_zeros = 0
    for name, param in model.named_parameters():
        if "weight" in name and ("fc" in name or "conv" in name):
        #if "conv" in name and "weight" in name:
            layer += 1
            num_tot = param.detach().cpu().numpy().size
            num_zeros = param.detach().cpu().eq(0).sum().item()
            sparsity = (num_zeros / num_tot) * 100
            density = 100 - sparsity
            tot_param += num_tot
            tot_zeros += num_zeros
            print("{}, {}, density: {:.2f}%, sparsity:{:.2f}%, total: {}, zeros: {}, non-zeros: {} ".format(layer, name,
                                                                                                          density,
                                                                                                          sparsity,
                                                                                                          num_tot,
                                                                                                          num_zeros,
                                                                                                          num_tot - num_zeros))

    print("Total parameters: {}, total zeros: {}, total non-zeros: {}".format(tot_param, tot_zeros, tot_param-tot_zeros))
    print("Total sparsity: {:.4f}, compression rate: {:.4f}".format(tot_zeros / tot_param,
                                                                    tot_param / (tot_param - tot_zeros)))

# --------------------------------------------------------------

def admm_adjust_learning_rate(args, optimizer, epoch):
    """ (The pytorch learning rate scheduler)
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by args.admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    lr = args.lr
    #lr = None
    #if epoch % args.admm_epoch == 0:
    #    lr = args.lr
    #else:
    #    args.admm_epoch_offset = epoch % args.admm_epoch

    #    admm_step = float(args.admm_epoch) / float(3)  # roughly every 1/3 args.admm_epoch.

    #    lr = args.lr * (0.1 ** (args.admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr