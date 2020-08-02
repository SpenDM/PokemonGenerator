class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Pokedex generator options")

        # Mode (required) 
        self.parser.add_argument("--mode",
                                 type=str,
                                 help="choose from [train, eval, predict]",
                                 required=True)
        
        # Data 
        self.parser.add_argument("--text_file",
                                 type=str,
                                 help="input text file for train or eval")
        self.parser.add_argument("--ckpt_file",
                                type=str,
                                default="distilgpt2"
                                help="location of model checkpoint, default load pretrained.")
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