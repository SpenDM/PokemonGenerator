# https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
import os
import math
import argparse

from config import Config
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



class PokedexGenerator():

    def __init__(self, opts):
        # Command line arguments
        self.opts = opts 

        # Load model and tokenizer
        config = AutoConfig.from_pretrained(opts.ckpt_file) 
        self.tokenizer = AutoTokenizer.from_pretrained(opts.ckpt_file)
        self.model = AutoModelWithLMHead.from_pretrained(opts.ckpt_file, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load training arguments
        if opts.mode == 'train' or opts.mode == 'eval':
            self.training_args = TrainingArguments
            self.training_args.device = 'cpu'
            self.training_args.n_gpu = 0
            self.training_args.logging_dir = opts.output_dir
            self.training_args.output_dir = opts.output_dir
            self.training_args.num_train_epochs = opts.num_epochs
            self.training_args.learning_rate = opts.learning_rate
            self.training_args.train_batch_size = opts.batch_size
            self.training_args.eval_batch_size = opts.batch_size
    
        # Load dataset 
        if opts.mode == 'train' or opts.mode == 'eval':
            self.dataset = LineByLineTextDataset( # TextDataset
                tokenizer=self.tokenizer, 
                file_path=opts.text_file, 
                block_size=self.tokenizer.max_len
            ) 
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False)

    def train(self):
        # Initialize trainer
        self.training_args.do_train = True
        trainer = Trainer(
            model=self.model, args=self.training_args,
            data_collator=self.data_collator, train_dataset=self.dataset,
            prediction_loss_only=True,
        )

        # Start training
        trainer.train()

        # Save trained model and tokenizer
        self.trainer.save_model() # TODO: check where this saves!
        self.tokenizer.save_pretrained(self.opts.output_dir)
    

    def eval(self):
        # Initialize trainer
        self.training_args.do_eval = True
        trainer = Trainer(
            model=self.model, args=self.training_args,
            data_collator=self.data_collator, eval_dataset=self.dataset,
            prediction_loss_only=True,
        )

        # Run evaluation
        eval_output = trainer.evaluate()

        # Summarize evaluation results in dict
        results = {}
        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(opts.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
        results.update(result)
    

    def predict(self):
        args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)

        prompt_text = args.prompt 

        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)

        return generated_sequences



if __name__ == "__main__":
    config = Config()
    opts = config.parse()
    g = PokdexGenerator(opts)
    if opts.mode == 'train':
        g.train()
    elif opts.mode == 'eval':
        g.eval()
    elif opts.mode == 'predict':
        g.predict()
    else: 
        raise ValueError('Specify valid <mode> = {train, eval, predict}.')