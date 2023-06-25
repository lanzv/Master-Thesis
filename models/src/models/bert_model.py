from transformers import BertTokenizer, LineByLineTextDataset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import wandb
import os
import torch
from transformers import BertTokenizer, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from src.models.bert.util import converter
from src.models.bert.SegmentBERT import SegmentBERT
from src.models.bert.trainer import distcriminative_train, generative_train
from src.models.bert.segmentor import seg
import os
import torch
from transformers import BertTokenizer, BertConfig
from src.eval.maww_score import mawp_score


class BERT_Model:
    def __init__(self, working_dir = "/kaggle/working", vocab_file = "vocab.txt", config_file = "config.json",
                 train_chants_file="train_chants.txt", dev_chants_file="dev_chants.txt", all_chants="chants.txt",
                 pretrained_dir="PretrainedBERT_chant/"):
        self.working_dir = working_dir
        self.vocab_file = vocab_file
        self.config_file = config_file
        self.train_chants_file = train_chants_file
        self.dev_chants_file = dev_chants_file
        self.all_chants = all_chants
        self.pretrained_dir = pretrained_dir
    
    def pretrain(self, epochs = 40, key="4ab08e6414f3948952d1d8bb5cce3be222d8ffd9"):
        wandb.login(key=key)
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.working_dir+"/"+self.vocab_file)
        bert_config = BertConfig.from_json_file(self.working_dir+"/"+self.config_file)

        train_dataset= LineByLineTextDataset(
            tokenizer = tokenizer,
            file_path = self.working_dir +"/"+ self.train_chants_file,
            block_size = 500  # maximum sequence length
        )
        dev_dataset= LineByLineTextDataset(
            tokenizer = tokenizer,
            file_path = self.working_dir +"/"+ self.dev_chants_file,
            block_size = 500  # maximum sequence length
        )

        model = BertForMaskedLM(bert_config)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir=self.working_dir+"/"+self.pretrained_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=32,
            save_total_limit=1,
            do_train = True,
            do_eval = True,
            overwrite_output_dir=True,
            evaluation_strategy="epoch", 
            logging_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset
        )
        trainer.train()
        trainer.save_model(self.working_dir+"/"+self.pretrained_dir)



    def train(self, num_epochs = 4, learning_epoch = 3200, border_I = 11.5, border_B = 8.6, prob_I_conf = -0.5, prob_B_conf = 0.5):
        #dataset = args.dataset # 'pku' or 'msr'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        output_dir = self.working_dir

        tokenizer = BertTokenizer.from_pretrained(self.working_dir+"/"+self.vocab_file)
        bert_config = BertConfig.from_json_file(self.working_dir+"/"+self.config_file)
        model = SegmentBERT(bert_config)
        state_dict = torch.load(f'{self.working_dir}/{self.pretrained_dir}pytorch_model.bin', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model.to(device=0)

        texts = []
        with open(self.working_dir+"/"+self.train_chants_file, 'r') as f:
            for line in f.readlines():
                single_chant = line.replace("\n", "")
                while len(single_chant) > 500:
                    texts.append(single_chant[:500])
                    single_chant = single_chant[500:]
                texts.append(single_chant)
        num_sample = len(texts)


        # train discriminative module which is randomly initialized
        num_epochs = 2
        num_training_steps = num_sample * num_epochs
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler_class = get_constant_schedule_with_warmup
        scheduler_args = {'num_warmup_steps':int(0.5*num_training_steps)}
        scheduler = scheduler_class(**{'optimizer':optimizer}, **scheduler_args)
        distcriminative_train(model, optimizer, num_sample, texts, tokenizer, -1, scheduler, start=0, end=num_training_steps, save_model=False, border_I = border_I, border_B = border_B)
        # save model
        coreModel = model.module if hasattr(model, "module") else model
        state_dict = coreModel.state_dict()
        torch.save(state_dict, os.path.join(output_dir, f"SegmentBERT_chants.pkl"))



        # iterative training
        num_epochs = num_epochs
        num_training_steps = num_sample * num_epochs
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler_class = get_linear_schedule_with_warmup
        #scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}
        scheduler_args = {'num_warmup_steps':int(0.0*num_training_steps), 'num_training_steps':num_training_steps}
        scheduler = scheduler_class(**{'optimizer':optimizer}, **scheduler_args)
        learning_epoch = learning_epoch
        n = (num_training_steps // learning_epoch) + 1
        for i in range(n):
            print("Iterative training: {} / {}".format(i + 1, n))
            generative_train(model, optimizer, num_sample, texts, tokenizer, scheduler, start=i * learning_epoch, end=i * learning_epoch + learning_epoch // 2, border_I = border_I, border_B = border_B, prob_I_conf = prob_I_conf, prob_B_conf = prob_B_conf)
            distcriminative_train(model, optimizer, num_sample, texts, tokenizer, scheduler, start=i * learning_epoch + learning_epoch // 2, end=(i + 1) * learning_epoch, border_I = border_I, border_B = border_B)

    def predict_segments(self, chants):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model_dir = self.working_dir

        cvt = converter(vocab_path=self.working_dir+"/"+self.vocab_file)
        segmented_chants = []
        tokenizer = BertTokenizer.from_pretrained(self.working_dir+"/"+self.vocab_file)
        bert_config = BertConfig.from_json_file(self.working_dir+"/"+self.config_file)
        model = SegmentBERT(bert_config)

        state_dict = torch.load(f'{model_dir}/SegmentBERT_chants.pkl', map_location='cpu')
        model.load_state_dict(state_dict)
        for line in chants:
            sentence = line.replace("\n", "")
            final_result = str()
            while len(sentence) > 500:
                result = seg(sentence[:500], model, tokenizer, cvt)
                sentence = sentence[500:]
                final_result += result
            result = seg(sentence, model, tokenizer, cvt)
            final_result += result
            segmented_chants.append(final_result.split(" "))

        return segmented_chants

    def get_mawp(self):
        return mawp_score(self)