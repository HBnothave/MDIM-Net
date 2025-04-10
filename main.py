from trainer import DiffusionTrainer

if __name__ == "__main__":
    trainer = DiffusionTrainer()
    trainer.run(epochs=100)