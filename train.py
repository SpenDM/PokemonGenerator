import os
import math

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
        self.parser.add_argument("--train_data_file",
                                 type=str,
                                 help="input training data file (a text file)")
        self.parser.add_argument("--eval_data_file",
                                 type=str,
                                 help="input evaluation data file (a text file)")
        self.parser.add_argument("--output_dir",
                                 type=str,
                                 help="directory where model predictions and checkpoints will be written.")
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
 
def main(cfg):

    MODEL_NAME = 'distilgpt2' # 'gpt2'

    # Training arguments
    training_args = TrainingArguments
    training_args.do_train = True
    training_args.do_eval = True

    # Load pretrained model and tokenizer
    opts = cfg.parse()
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelWithLMHead.from_pretrained(MODEL_NAME, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Get datasets
    train_dataset = LineByLineTextDataset( # TextDataset?
        tokenizer=tokenizer, 
        file_path=opts.train_data_file, 
        block_size=tokenizer.max_len
    ) 
    eval_dataset = LineByLineTextDataset( # TextDataset?
        tokenizer=tokenizer, 
        file_path=opts.eval_data_file, 
        block_size=tokenizer.max_len
    ) 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(opts.output_dir)

    # Evaluation
    results = {}
    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(opts.output_dir, "eval_results_lm.txt")
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    results.update(result)

    return results

if __name__ == "__main__":
    main()