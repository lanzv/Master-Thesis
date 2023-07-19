import wandb
import os
import torch
import os
import torch
from src.eval.maww_score import mawp_score


class BERT_Model:
    def __init__(self, working_dir = ".", vocab_file = "vocab.txt", config_file = "config.json",
                 train_chants_file="train_chants.txt", dev_chants_file="dev_chants.txt", all_chants="chants.txt",
                 pretrained_dir="PretrainedBERT_chant/"):
        """
        Initialize the BERT model

        Parameters
        ----------
        working_dir: string
            working directory path, "." for the root, "/kaggle/working" for the kaggle workspace
        vocab_file: string
            path to the vocab.txt file, by default it is placed in root
        config_file: string
            path to the config.json file, by default it is placed in root
        train_chants_file: string
            chants training dataset of melodies as string in text file, each line consists of one string melody (one chant)
        dev_chants_file: string
            chants dev dataset of melodies as string in text file, each line consists of one string melody (one chant)
        all_chants: string
            chants (train+dev+test) dataset of melodies as string in text file, each line consists of one string melody (one chant)
        pretrained_dir: string
            path to the directory that the pretrained BERT model is/will be saved in
        """
        self.working_dir = working_dir
        self.vocab_file = vocab_file
        self.config_file = config_file
        self.train_chants_file = train_chants_file
        self.dev_chants_file = dev_chants_file
        self.all_chants = all_chants
        self.pretrained_dir = pretrained_dir
    
    def pretrain(self, epochs = 40, key="4ab08e6414f3948952d1d8bb5cce3be222d8ffd9"):
        """
        Pretrain the BERT model for the chant segmentation task
        The pretrained model will be stored into folder of self.pretrained_dir
        We need the pytorch_model.bin file that the BERT training uses as the initial state values

        Parameters
        ----------
        epochs: int
            number of training epochs
        key: string
            wandb token to see the model training statistics
        """
        wandb.login(key=key)
        from transformers import BertTokenizer, LineByLineTextDataset
        from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
        from transformers import Trainer, TrainingArguments
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
        """
        Parametrized training procedure of BERT designed by QbethQ (https://github.com/QbethQ/Unsupervised_CWS_BOPT)
        The pretrained BERT has to be done first
        The trained BERT model is saved to SegmentBERT_chants.pkl file in the self.working_dir
        
        Parameters
        ----------
        num_epochs: int
            number of generative-discriminative stage epochs
        learning_epoch: int
            number of chants being part of one generative-discriminative iteration
        border_I: float
            threshold of tone being inside the segment
        border_B: float
            threshold of tone being beginning of the segment
        prob_I_conf: float
            threshold of the softmax I prediction confidential
        prob_B_conf: float
            threshold of the softmax B prediction confidential
        """
        from src.models.bert.SegmentBERT import SegmentBERT
        from src.models.bert.trainer import distcriminative_train, generative_train
        from transformers import BertTokenizer, BertConfig, AdamW
        from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
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
        distcriminative_train(model, optimizer, num_sample, texts, tokenizer, scheduler, start=0, end=num_training_steps, save_model=False, border_I = border_I, border_B = border_B)
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
        """
        Predicting procedure of BERT designed by QbethQ (https://github.com/QbethQ/Unsupervised_CWS_BOPT)
        The BERT has to be trained first, stored in file SegmentBERT_chants.pkl in the self.working_dir directory

        
        Parameters
        ----------
        chants: list of strings
            list of melody strings
        Returns
        -------
        segmented_chants: list of lists of strings
            list of segmented chants
        perplexity: int, -1
            perplexity cannot be computed, -1 for the model generality that the
            predict_segments() function could be run generally from other scripts
        """
        
        from src.models.bert.util import converter
        from src.models.bert.SegmentBERT import SegmentBERT
        from src.models.bert.segmentor import seg
        from transformers import BertTokenizer, BertConfig
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

        return segmented_chants, -1 #-1 for unknown perplexity

    def get_mawp_score(self):
        """
        Get the mawp score of the BERT model

        Returns
        -------
        mawp: float
            mawp score
        """
        return mawp_score(self)