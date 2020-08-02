import os
import math
import argparse

from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    TextDataset,
    Trainer,
    TrainingArguments
)


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Pokedex generator options")

        # Data 
        self.parser.add_argument("--mode",
                                 type=str,
                                 help="choose from [train, eval, predict]")
        self.parser.add_argument("--data_file",
                                 type=str,
                                 help="input data file (a text file)")
        self.parser.add_argument("--output_dir",
                                 type=str,
                                 help="directory where model predictions and checkpoints will be written.")
        
        # Train
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 default=3,
                                 help="number of epochs to train")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 default=5e-5,
                                 help="initial learning rate for optimizer")
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 default=8,
                                 help="number to include in batch")
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
 
def main(opts):

    MODEL_NAME = 'distilgpt2' # 'gpt2'

    # Training arguments
    training_args = TrainingArguments
    training_args.device = 'cpu'
    training_args.n_gpu = 0
    training_args.logging_dir = opts.output_dir
    training_args.output_dir = opts.output_dir
    training_args.num_train_epochs = opts.num_epochs
    training_args.learning_rate = opts.learning_rate
    training_args.train_batch_size = opts.batch_size
    training_args.eval_batch_size = opts.batch_size

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelWithLMHead.from_pretrained(MODEL_NAME, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Get datasets
    dataset = LineByLineTextDataset( # TextDataset
        tokenizer=tokenizer, 
        file_path=opts.data_file, 
        block_size=tokenizer.max_len
    ) 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Mode
    if opts.mode == 'train':
        training_args.do_train = True
        trainer = Trainer(
            model=model, args=training_args,
            data_collator=data_collator, train_dataset=dataset,
            prediction_loss_only=True,
        )
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(opts.output_dir)
    
    # Evaluation Mode
    elif opts.mode == 'eval':
        training_args.do_eval = True
        trainer = Trainer(
            model=model, args=training_args,
            data_collator=data_collator, eval_dataset=dataset,
            prediction_loss_only=True,
        )

        results = {}
        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(opts.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        results.update(result)

    else:
        raise ValueError('Specify valid mode, either train or eval')

  
    

if __name__ == "__main__":
    config = Config()
    opts = config.parse()
    main(opts)