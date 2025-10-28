"""
Main training script orchestrating all phases
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import argparse
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from models.attribute_ranker import MultiAttributeRanker, AttributeRanker
from models.speaker_encoder import SpeakerEmbeddingExtractor

from data.dataset import (
    PairwiseRankingDataset,
    GANTrainingDataset
)
from training.ranker_trainer import SingleAttributeRankerTrainer
from training.gan_trainer import RelativeAttributeGAN
from evaluation.evaluator  import GANEvaluator


def train_attribute_rankers(args):
    """Phase 1: Train attribute ranking networks"""
    print("\n" + "="*60)
    print("PHASE 1: Training Attribute Rankers")
    print("="*60)
    
    # Load data
    # embeddings = torch.load(args.embeddings_path)
    metadata = pd.read_csv(args.metadata_path)
    
    # Create datasets for each attribute
    for attribute in args.attributes:
        print(f"\n{'='*60}")
        print(f"Training ranker for attribute: {attribute}")
        print(f"{'='*60}")
        
        dataset = PairwiseRankingDataset(
            metadata, attribute,
            num_pairs_per_sample=args.pairs_per_sample
        )
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # # Create model
        # model = AttributeRankerTrainer(
        #     embedding_dim=args.embedding_dim,
        #     attributes=[attribute],
        #     learning_rate=args.ranker_lr
        # )

        # Create model for THIS attribute
        model = SingleAttributeRankerTrainer(
            attribute_name=attribute,
            embedding_dim=args.embedding_dim,
            learning_rate=args.ranker_lr
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            dirpath=f'checkpoints/rankers/{attribute}',
            filename='best',
            monitor=f'val_loss',
            mode='min',
            save_top_k=1
        )
        
        early_stop = EarlyStopping(
            monitor=f'val_loss',
            patience=10,
            mode='min'
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=args.ranker_epochs,
            callbacks=[checkpoint, early_stop],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=WandbLogger(project='speaker-gan-rankers', name=f'ranker_{attribute}')
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        print(f"✓ Completed training for {attribute}")
    
    print("\n✓ Phase 1 Complete: All attribute rankers trained")



def train_gan(args):
    """Phase 2-3: GAN training with progressive curriculum"""
    print("\n" + "="*60)
    print("PHASE 2-3: Training GAN with Progressive Curriculum")
    print("="*60)
    
    # Load data
    # embeddings = torch.load(args.embeddings_path)
    metadata = pd.read_csv(args.metadata_path)
    
    # Create dataset
    dataset = GANTrainingDataset(
        metadata,
        attributes=args.attributes
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.gan_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.gan_batch_size,
        num_workers=args.num_workers
    )
    
    # Create GAN model with ranker checkpoint path
    model = RelativeAttributeGAN(
        embedding_dim=args.embedding_dim,
        num_attributes=len(args.attributes),
        attributes=args.attributes,  # NEW: Pass attribute names
        generator_hidden_dim=args.generator_hidden_dim,
        discriminator_hidden_dim=args.discriminator_hidden_dim,
        lr_g=args.gan_lr_g,
        lr_d=args.gan_lr_d,
        lambda_attr=args.lambda_attr,
        lambda_dist=args.lambda_dist,
        lambda_smooth=args.lambda_smooth,
        lambda_ranker=args.lambda_ranker,  # NEW
        ranker_checkpoint_dir='checkpoints/rankers'  # NEW: Load pretrained rankers
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        dirpath='checkpoints/gan',
        filename='epoch_{epoch:03d}',
        monitor='g_loss_total',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=5
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.gan_epochs,
        callbacks=[checkpoint],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=WandbLogger(project='speaker-gan', name='relative_attr_gan'),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.25
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✓ Phase 2-3 Complete: GAN training finished")


def evaluate(args):
    """Phase 4: Comprehensive evaluation - CORRECTED VERSION"""
    print("\n" + "="*60)
    print("PHASE 4: Evaluation")
    print("="*60)
    
    # Load trained models
    gan = RelativeAttributeGAN.load_from_checkpoint(args.gan_checkpoint)
    gan.eval()
    
    # Load attribute rankers
    rankers = MultiAttributeRanker(
        embedding_dim=args.embedding_dim,
        attributes=args.attributes
    )
    for attr in args.attributes:
        ranker_ckpt = torch.load(f'checkpoints/rankers/{attr}/best.ckpt')
        # Extract ranker state dict from Lightning checkpoint
        state_dict = {}
        for key, value in ranker_ckpt['state_dict'].items():
            if key.startswith(f'ranker.{attr}'):
                new_key = key.replace(f'ranker.{attr}.', '')
                state_dict[new_key] = value
        rankers.rankers[attr].load_state_dict(state_dict, strict=False)
    
    # Load speaker encoder (optional but useful for analysis)
    encoder = None
    # if args.encoder_path:
    #     encoder = SpeakerEmbeddingExtractor(pretrained_path=args.encoder_path)
    
    # Load test data
    # test_embeddings = torch.load(args.test_embeddings_path)
    test_metadata = pd.read_csv(args.test_metadata_path)
    embedding_paths = test_metadata['embedding_path'].values
    test_embeddings = []
    for path in embedding_paths:
        emb = np.load(path)
        emb_tensor = torch.from_numpy(emb)
        test_embeddings.append(emb_tensor)
    test_embeddings = torch.stack(test_embeddings)
    
    # Create evaluator
    evaluator = GANEvaluator(
        generator=gan.generator,
        discriminator=gan.discriminator,
        attribute_rankers=rankers,
        speaker_encoder=encoder
    )
    
    # Run evaluation
    lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = evaluator.evaluate_all(
        test_embeddings,
        test_metadata,
        lambda_values=lambda_values
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    import json
    results_path = 'evaluation_results.json'
    
    # Convert numpy types to native Python for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("\n✓ Evaluation Complete")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    
    # General
    parser.add_argument('--mode', type=str, choices=['ranker', 'gan', 'eval', 'all'],
                       default='all', help='Training mode')
    # parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, default='/data/waris/data/Voxceleb/voxceleb1/final_metadata_train.csv')
    # parser.add_argument('--test_embeddings_path', type=str)
    parser.add_argument('--test_metadata_path', type=str, default='/data/waris/data/Voxceleb/voxceleb1/final_metadata_test.csv')
    # parser.add_argument('--encoder_path', type=str, help='Pretrained speaker encoder')
    
    # Data
    parser.add_argument('--embedding_dim', type=int, default=704)
    parser.add_argument('--attributes', nargs='+',
                       default=['age', 'gender', 'pitch', 'voice_quality'])
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Ranker training
    parser.add_argument('--ranker_epochs', type=int, default=50)
    parser.add_argument('--ranker_lr', type=float, default=1e-3)
    parser.add_argument('--pairs_per_sample', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    
    # GAN training
    parser.add_argument('--gan_epochs', type=int, default=100)
    parser.add_argument('--gan_batch_size', type=int, default=64)
    parser.add_argument('--gan_lr_g', type=float, default=1e-4)
    parser.add_argument('--gan_lr_d', type=float, default=4e-4)
    parser.add_argument('--generator_hidden_dim', type=int, default=512)
    parser.add_argument('--discriminator_hidden_dim', type=int, default=512)
    parser.add_argument('--lambda_attr', type=float, default=1.0)
    parser.add_argument('--lambda_dist', type=float, default=1.0)
    parser.add_argument('--lambda_smooth', type=float, default=0.1)
    
    # Evaluation
    parser.add_argument('--gan_checkpoint', type=str)
    
    args = parser.parse_args()
    
    # Execute requested mode
    if args.mode in ['ranker', 'all']:
        train_attribute_rankers(args)
    
    if args.mode in ['gan', 'all']:
        train_gan(args)
    
    if args.mode in ['eval', 'all']:
        if not args.gan_checkpoint:
            args.gan_checkpoint = 'checkpoints/gan/last.ckpt'
        evaluate(args)


if __name__ == '__main__':
    main()
