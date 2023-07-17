import os
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random



def dist(x, y):
    '''
    distance function
    '''
    return np.sqrt(((x - y)**2).sum())

def distcriminative_train(model, optimizer, num_sample, texts, tokenizer, scheduler, start=0, end=0, save_model=True, border_I = 11.5, border_B = 8.6):
    '''
    discriminative module learns from generative module
    '''
    optimizer.zero_grad()
    masked_lm_loss = 0.0
    discriminative_loss = 0.0
    for epoch in tqdm(range(start, end), unit="epoch", mininterval=10):
        sentence = texts[epoch % num_sample]
        input_ids = []
        i = 0
        while i < len(sentence):
            input_ids += tokenizer.encode(sentence[i:(i+100)], add_special_tokens=False)
            i += 100
        length = len(input_ids)
        input_ids.insert(0, 1) # id of [CLS]
        input_ids.insert(length + 1, 2) # id of [SEP]

        # random mask to count masked_lm_loss
        masked_lm_input_ids = torch.tensor(input_ids)
        masked_lm_labels = torch.tensor(input_ids)
        for i in range(1, length + 1):
            r = random.random()
            if r < 0.12:
                masked_lm_input_ids[i] = 3 # id of [mask]
        masked_lm_loss += (model(input_ids=masked_lm_input_ids.unsqueeze(0).to(0), masked_lm_labels=masked_lm_labels.unsqueeze(0).to(0)))[0].cpu()	

        # count discriminative_loss
        logits = (model(input_ids=torch.tensor(input_ids).unsqueeze(0).to(0), mode=1))[0].cpu()

        with torch.no_grad():
            ninput_ids = np.array([input_ids] * (2 * length - 1))

            for i in range(length):
                if i > 0:
                    ninput_ids[2 * i - 1, i] = 3 # id of [mask]
                    ninput_ids[2 * i - 1, i + 1] = 3 # id of [mask]
                ninput_ids[2 * i, i + 1] = 3 # id of [mask]

            batch_size = 16
            batch_num = ninput_ids.shape[0] // batch_size if ninput_ids.shape[0] % batch_size == 0 else (ninput_ids.shape[0] // batch_size) + 1
            small_batches = [[ninput_ids[num*batch_size : (num+1)*batch_size]] for num in range(batch_num)]
            for num, [input] in enumerate(small_batches):
                if num == 0:
                    vectors = (model((torch.from_numpy(input)).to(0)))[0].cpu().detach().numpy()	
                else:
                    tmp_vectors = (model((torch.from_numpy(input)).to(0)))[0].cpu().detach().numpy()
                    vectors = np.concatenate((vectors, tmp_vectors), axis=0)
            labels = []
            labels.append(0) # [CLS]
            labels.append(0) # first character

            for i in range(1, length): # decide whether the i-th character and the (i+1)-th character should be in one word
                d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
                d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
                d = (d1 + d2) / 2

                if d >= border_I:
                    labels.append(1)
                elif d >= border_B:
                    labels.append(-100) # -100 is ignored in CrossEntropyLoss()
                else:
                    labels.append(0)
            labels.append(0) # [SEP]

        loss_fct = CrossEntropyLoss()
        discriminative_loss += loss_fct(logits.view(-1, 2), torch.tensor(labels).view(-1))

        # count total loss and backward
        if ((epoch + 1) % 32 == 0) or (epoch == end - 1):
            k_m = 0.30
            k_d = 1 - k_m
            loss = k_m * masked_lm_loss + k_d * discriminative_loss
            print("epoch {}:\n\t{:.1f} * masked_lm_loss = {:.4f}\n\t{:.1f} * discriminative_loss = {:.4f}\n\ttotal loss = {:.4f}".format(epoch + 1, k_m, k_m * masked_lm_loss, k_d, k_d * discriminative_loss, loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            masked_lm_loss = 0.0
            discriminative_loss = 0.0

    # save model
    if save_model:
        coreModel = model.module if hasattr(model, "module") else model
        state_dict = coreModel.state_dict()
        torch.save(state_dict, os.path.join(".", f"SegmentBERT_chants.pkl"))

def generative_train(model, optimizer, num_sample, texts, tokenizer, scheduler, start=0, end=0, save_model=False, border_I = 11.5, border_B = 8.6, prob_I_conf = -0.5, prob_B_conf = 0.5):
    '''
    generative module learns from discriminative module
    '''
    optimizer.zero_grad()
    masked_lm_loss = 0.0
    generative_loss = 0.0
    for epoch in tqdm(range(start, end), unit="epoch", mininterval=10):
        sentence = texts[epoch % num_sample]
        input_ids = []
        i = 0
        while i < len(sentence):
            input_ids += tokenizer.encode(sentence[i:(i+100)], add_special_tokens=False)
            i += 100
        length = len(input_ids)
        input_ids.insert(0, 1) # id of [CLS]
        input_ids.insert(length + 1, 2) # id of [SEP]

        # random mask to count masked_lm_loss
        masked_lm_input_ids = torch.tensor(input_ids)
        masked_lm_labels = torch.tensor(input_ids)
        for i in range(1, length + 1):
            r = random.random()
            if r < 0.12:
                masked_lm_input_ids[i] = 3 # id of [mask]
        masked_lm_loss += (model(input_ids=masked_lm_input_ids.unsqueeze(0).to(0), masked_lm_labels=masked_lm_labels.unsqueeze(0).to(0)))[0].cpu()
        # count generative_loss
        with torch.no_grad():
            logits = (model(input_ids=torch.tensor(input_ids).unsqueeze(0).to(0), mode=1))[0][0].cpu()
            probs = F.softmax(logits, dim=1)[1:length + 1]
            del logits
            probs = probs.t()
            prob_score = probs[0] - probs[1] # P(label 0) - P(label 1)
            del probs

        ninput_ids = np.array([input_ids] * (2 * length - 1))

        for i in range(length):
            if i > 0:
                ninput_ids[2 * i - 1, i] = 3 # id of [mask]
                ninput_ids[2 * i - 1, i + 1] = 3 # id of [mask]
            ninput_ids[2 * i, i + 1] = 3 # id of [mask]

        batch_size = 16
        batch_num = ninput_ids.shape[0] // batch_size if ninput_ids.shape[0] % batch_size == 0 else (ninput_ids.shape[0] // batch_size) + 1
        small_batches = [[ninput_ids[num*batch_size : (num+1)*batch_size]] for num in range(batch_num)]
        for num, [input] in enumerate(small_batches):
            if num == 0:
                vectors = (model((torch.from_numpy(input)).to(0)))[0].cpu().detach().numpy()
            else:
                tmp_vectors = (model((torch.from_numpy(input)).to(0)))[0].cpu().detach().numpy()
                vectors = np.concatenate((vectors, tmp_vectors), axis=0)

        d_lst = []
        target_lst = []
        for i in range(1, length):
            d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
            d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
            d = (d1 + d2) / 2
            d_lst.append(d)
            if prob_score[i] >= prob_B_conf and d > border_B:
                target_lst.append(border_B)
            elif prob_score[i] <= prob_I_conf and d < border_I:
                target_lst.append(border_I)
            else:
                target_lst.append(d)
        loss_fct = MSELoss()
        generative_loss += loss_fct(torch.tensor(d_lst), torch.tensor(target_lst))

        # count total loss and backward
        if ((epoch + 1) % 32 == 0) or (epoch == end - 1):
            k_m = 0.80
            k_g = 1.00 - k_m
            loss = k_m * masked_lm_loss + k_g * generative_loss
            print("epoch {}:\n\t{:.1f} * masked_lm_loss = {:.4f}\n\t{:.1f} * generative_loss = {:.4f}\n\ttotal loss = {:.4f}".format(epoch + 1, k_m, k_m * masked_lm_loss, k_g, k_g * generative_loss, loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            masked_lm_loss = 0.0
            generative_loss = 0.0

    # save model
    if save_model:
        coreModel = model.module if hasattr(model, "module") else model
        state_dict = coreModel.state_dict()
        torch.save(state_dict, os.path.join(".", f"SegmentBERT_chants.pkl"))
