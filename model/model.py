import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DNASequenceRNN(nn.Module):
    """
    A recurrent neural network model for generating DNA sequences.
    """
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.2):
        super(DNASequenceRNN, self).__init__()
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '<START>': 4, '<END>': 5, '<UNK>': 6}
        self.idx_to_char = {i: c for c, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        
        # Loss prediction network
        self.loss_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, hidden=None):
        # Convert input to embeddings
        x = self.embedding(x)
        
        # Pass through LSTM
        output, hidden = self.lstm(x, hidden)
        
        # Pass through final layer
        sequence_output = self.fc(output)
        
        return sequence_output, hidden, output
    
    def predict_loss(self, lstm_output):
        # Use the final hidden state to predict the loss
        final_hidden = lstm_output[:, -1, :]
        return self.loss_predictor(final_hidden)
    
    def init_hidden(self, batch_size=1):
        # Initialize hidden state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class DNASequenceGenerator:
    """
    A generator for DNA sequences that uses a PyTorch model and reinforcement learning
    to generate sequences with minimal loss.
    """
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.2, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DNASequenceRNN(hidden_size, num_layers, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.sequence_criterion = nn.CrossEntropyLoss()
        self.loss_criterion = nn.MSELoss()
        
        # Track best sequence and its loss
        self.best_sequence = None
        self.best_loss = float('inf')
        self.historical_losses = []
        self.historical_sequences = []
        self.trinucleotide_probs = {}
        
    def _sequence_to_tensor(self, sequence):
        """Convert a DNA sequence to a tensor of indices"""
        if sequence is None:
            return torch.tensor([[self.model.vocab['<START>'], self.model.vocab['<END>']]], 
                                dtype=torch.long, device=self.device)
        
        start_token = self.model.vocab['<START>']
        end_token = self.model.vocab['<END>']
        
        # Handle None values and non-ACGT characters
        sequence_indices = [start_token]
        for nucl in sequence:
            if nucl in self.model.vocab:
                sequence_indices.append(self.model.vocab[nucl])
            else:
                # Use unknown token for any character not in vocabulary
                sequence_indices.append(self.model.vocab['<UNK>'])
        sequence_indices.append(end_token)
        
        return torch.tensor([sequence_indices], dtype=torch.long, device=self.device)
    
    def _tensor_to_sequence(self, tensor):
        """Convert a tensor of indices back to a DNA sequence"""
        indices = tensor.squeeze().tolist()
        # Handle the case where tensor is a single number
        if isinstance(indices, int):
            indices = [indices]
            
        # Remove start, end, and unknown tokens
        filtered_indices = []
        for idx in indices:
            if idx < 4:  # Only include A, C, G, T
                filtered_indices.append(idx)
        
        return ''.join(self.model.idx_to_char[idx] for idx in filtered_indices)
    
    def _build_trinucleotide_model(self):
        """
        Build a model of trinucleotide frequencies weighted by inverse loss
        from the training history.
        """
        self.trinucleotide_probs = {}
        
        # If no training data, return empty model
        if not self.historical_losses or not self.historical_sequences:
            print("Warning: No training data available to build trinucleotide model")
            return
        
        # Calculate weights based on inverse loss (higher weight for lower loss)
        max_loss = max(self.historical_losses)
        min_loss = min(self.historical_losses)
        loss_range = max_loss - min_loss
        if loss_range == 0:
            # If all losses are the same, use equal weights
            weights = [1.0] * len(self.historical_losses)
        else:
            # Scale weights to give higher values to lower losses
            weights = [max(0.1, (max_loss - loss) / loss_range) for loss in self.historical_losses]
        
        # Count trinucleotide frequencies weighted by inverse loss
        for sequence, weight in zip(self.historical_sequences, weights):
            if len(sequence) < 3:
                continue
                
            for i in range(len(sequence) - 2):
                trinucleotide = sequence[i:i+3]
                if trinucleotide not in self.trinucleotide_probs:
                    self.trinucleotide_probs[trinucleotide] = 0
                self.trinucleotide_probs[trinucleotide] += weight
        
        # Normalize probabilities
        sum_probs = sum(self.trinucleotide_probs.values())
        if sum_probs > 0:
            for trinucleotide in self.trinucleotide_probs:
                self.trinucleotide_probs[trinucleotide] /= sum_probs
    
    def train(self, sequence, loss):
        """
        Train the model on a given sequence and its loss.
        
        Args:
            sequence: A DNA sequence string (containing A, C, G, T)
            loss: The loss value for this sequence (lower is better)
        """
        # Handle None or empty sequence
        if sequence is None or len(sequence) == 0:
            print("Warning: Empty or None sequence provided to train(). Skipping.")
            return 0, 0
        
        # Check if sequence contains valid nucleotides
        valid_nucleotides = set('ACGT')
        if not all(nucl in valid_nucleotides for nucl in sequence):
            print(f"Warning: Sequence contains non-ACGT characters")
            # Clean the sequence
            sequence = ''.join(nucl for nucl in sequence if nucl in valid_nucleotides)
            if len(sequence) == 0:
                print("Sequence has no valid nucleotides after cleaning. Skipping.")
                return 0, 0
        
        # Track best sequence
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_sequence = sequence
        
        self.historical_losses.append(loss)
        self.historical_sequences.append(sequence)
        
        # Rebuild trinucleotide model periodically
        if len(self.historical_sequences) % 10 == 0:
            self._build_trinucleotide_model()
        
        # Convert sequence to tensor
        input_tensor = self._sequence_to_tensor(sequence)
        
        # Check if sequence is too short
        if input_tensor.size(1) < 3:  # START + at least one nucleotide + END
            print("Warning: Sequence too short for training. Skipping.")
            return 0, 0
        
        # Create target tensor (shifted by one position)
        target_tensor = input_tensor.clone()
        target_tensor = target_tensor[:, 1:]  # Remove first character (start token)
        
        # Input is all but the last character
        input_tensor = input_tensor[:, :-1]
        
        # Training using standard sequence prediction
        self.optimizer.zero_grad()
        self.model.train()
        
        try:
            # Forward pass
            sequence_output, _, lstm_output = self.model(input_tensor)
            
            # Predict loss
            predicted_loss = self.model.predict_loss(lstm_output)
            loss_tensor = torch.tensor([[loss]], dtype=torch.float, device=self.device)
            
            # Calculate sequence prediction loss
            seq_loss = self.sequence_criterion(sequence_output.reshape(-1, self.model.vocab_size), target_tensor.reshape(-1))
            
            # Calculate loss prediction loss
            loss_prediction_loss = self.loss_criterion(predicted_loss, loss_tensor)
            
            # Combine losses
            combined_loss = seq_loss + 0.1 * loss_prediction_loss
            
            # Add reinforcement learning component
            reward = torch.exp(torch.tensor(-loss))
            combined_loss = combined_loss * (1.0 - reward * 0.5)  # Scale reward influence
            
            combined_loss.backward()
            self.optimizer.step()
            
            return seq_loss.item(), loss_prediction_loss.item()
            
        except Exception as e:
            return 0, 0
    
    def generate(self, min_length=2000, max_length=2000000):
        """
        Generate a new DNA sequence based on trinucleotide frequencies weighted by inverse loss.
        
        Args:
            min_length: Minimum length of the generated sequence
            max_length: Maximum length of the generated sequence
                
        Returns:
            A tuple of (generated_sequence, estimated_loss)
        """
        # Make sure trinucleotide model is built
        if not self.trinucleotide_probs:
            self._build_trinucleotide_model()
        
        # If still no model (e.g., no training data yet), fall back to random generation
        if not self.trinucleotide_probs:
            print("Warning: No trinucleotide model available. Generating random sequence.")
            sequence = ''.join(random.choices(['A', 'C', 'G', 'T'], k=random.randint(min_length, max_length)))
            estimated_loss = self.estimate_loss(sequence)
            return sequence, estimated_loss
        
        # Choose a target length
        target_length = random.randint(min_length, max_length)
        
        # Start with a random dinucleotide
        # Get all possible starting dinucleotides
        possible_starts = list(set(k[:2] for k in self.trinucleotide_probs.keys()))
        if possible_starts:
            current = random.choice(possible_starts)
        else:
            # Fallback to random dinucleotide
            current = ''.join(random.choices(['A', 'C', 'G', 'T'], k=2))
        
        # Generate sequence
        sequence = current
        while len(sequence) < target_length:
            # Get possible next nucleotides based on current dinucleotide context
            context = sequence[-2:]
            possible_trinucleotides = [tri for tri in self.trinucleotide_probs.keys() if tri.startswith(context)]
            
            if possible_trinucleotides:
                # Get probabilities for possible trinucleotides
                probs = [self.trinucleotide_probs[tri] for tri in possible_trinucleotides]
                
                # Normalize probabilities
                sum_probs = sum(probs)
                if sum_probs > 0:
                    probs = [p / sum_probs for p in probs]
                    # Choose next nucleotide based on probabilities
                    chosen_tri = random.choices(possible_trinucleotides, weights=probs, k=1)[0]
                    next_nucleotide = chosen_tri[2]
                else:
                    # If all probabilities are zero, choose randomly
                    next_nucleotide = random.choice(['A', 'C', 'G', 'T'])
            else:
                # If no context found, choose randomly
                next_nucleotide = random.choice(['A', 'C', 'G', 'T'])
            
            # Add nucleotide to sequence
            sequence += next_nucleotide
        
        # Estimate loss for the generated sequence
        estimated_loss = self.estimate_loss(sequence)
        
        return sequence, estimated_loss
    
    def generate_old(self, min_length=2000, max_length=2000000, num_candidates=5):
        """
        Original generate function using RNN.
        
        Args:
            min_length: Minimum length of the generated sequence
            max_length: Maximum length of the generated sequence
            num_candidates: Number of candidate sequences to generate
            
        Returns:
            A tuple of (best_sequence, estimated_loss)
        """
        self.model.eval()
        
        best_candidate = None
        best_estimated_loss = float('inf')
        
        attempts = 0
        max_attempts = num_candidates * 2  # Allow some failed attempts
        
        # Generate multiple candidates and pick the best one
        while attempts < max_attempts and len([c for c in range(num_candidates) if best_candidate is not None]) < num_candidates:
            attempts += 1
            
            # Choose a target length
            target_length = random.randint(min_length, max_length)
            
            with torch.no_grad():
                # Start with start token
                input_tensor = torch.tensor([[self.model.vocab['<START>']]], device=self.device)
                hidden = self.model.init_hidden()
                
                # For storing entire sequence for loss prediction
                sequence_tensors = [input_tensor]
                
                # Generate sequence
                generated_sequence = []
                for i in range(max_length):
                    try:
                        output, hidden, _ = self.model(input_tensor, hidden)
                        
                        # Sample from the output distribution
                        probs = torch.softmax(output[0, -1], dim=0)
                        
                        # If we're at or above the target length, increase probability of END token
                        if i >= target_length - 1:
                            probs[self.model.vocab['<END>']] += 0.1 * (i - target_length + 1)
                            probs = probs / probs.sum()  # Renormalize
                        
                        # Ensure we only generate A, C, G, T (first 4 indices)
                        for j in range(4, len(probs)):
                            if j != self.model.vocab['<END>']:  # Keep END token probability
                                probs[j] = 0
                        
                        if probs.sum() > 0:
                            probs = probs / probs.sum()  # Renormalize
                        else:
                            # If all probabilities are zero, use uniform distribution for ACGT
                            probs = torch.zeros_like(probs)
                            for j in range(4):  # A, C, G, T
                                probs[j] = 0.25
                        
                        # Sample next character
                        next_char_idx = torch.multinomial(probs, 1).item()
                        
                        # Store the tensor for loss prediction
                        next_tensor = torch.tensor([[next_char_idx]], device=self.device)
                        sequence_tensors.append(next_tensor)
                        
                        # If we reached the end token, stop
                        if next_char_idx == self.model.vocab['<END>']:
                            break
                        
                        # Add to generated sequence if it's a nucleotide
                        if next_char_idx < 4:  # A, C, G, T
                            generated_sequence.append(self.model.idx_to_char[next_char_idx])
                        
                        # Update input for next iteration
                        input_tensor = next_tensor
                        
                    except Exception as e:
                        print(f"Error during sequence generation")
                        break
                
                # Convert sequence to string
                sequence_str = ''.join(generated_sequence)
                
                # If sequence is too short, skip
                if len(sequence_str) < min_length:
                    continue
                
                try:
                    # Estimate loss for the generated sequence
                    full_sequence_tensor = torch.cat(sequence_tensors, dim=1)
                    _, _, lstm_output = self.model(full_sequence_tensor)
                    estimated_loss = self.model.predict_loss(lstm_output).item()
                    
                    # Update best candidate if this one is better
                    if estimated_loss < best_estimated_loss:
                        best_estimated_loss = estimated_loss
                        best_candidate = sequence_str
                except Exception as e:
                    print(f"Error estimating loss for generated sequence: {e}")
                    continue
        
        if best_candidate is None:
            print("Warning: Failed to generate any valid sequences.")
            # Return a random sequence as fallback
            best_candidate = ''.join(random.choices(['A', 'C', 'G', 'T'], k=min_length))
            best_estimated_loss = float('inf')
            
        return best_candidate, best_estimated_loss
    
    def estimate_loss(self, sequence):
        """
        Estimate the loss for a given sequence using the loss predictor network.
        
        Args:
            sequence: A DNA sequence string
            
        Returns:
            Estimated loss value
        """
        self.model.eval()
        
        # Handle None or empty sequence
        if sequence is None or len(sequence) == 0:
            print("Warning: Empty or None sequence provided to estimate_loss()")
            return float('inf')
        
        # Clean the sequence to ensure only ACGT characters
        valid_nucleotides = set('ACGT')
        cleaned_sequence = ''.join(nucl for nucl in sequence if nucl in valid_nucleotides)
        
        try:
            with torch.no_grad():
                # Convert sequence to tensor
                input_tensor = self._sequence_to_tensor(cleaned_sequence)
                
                # Get LSTM output
                _, _, lstm_output = self.model(input_tensor)
                
                # Predict loss
                estimated_loss = self.model.predict_loss(lstm_output).item()
            
            return estimated_loss
        except Exception as e:
            print(f"Error in loss estimation: {e}")
            return float('inf')
    
    def get_stats(self):
        """Return statistics about the model's performance"""
        stats = {
            "best_loss": self.best_loss,
            "training_examples": len(self.historical_losses),
            "recent_losses": self.historical_losses[-10:] if self.historical_losses else [],
            "average_recent_loss": sum(self.historical_losses[-20:]) / len(self.historical_losses[-20:]) if len(self.historical_losses) >= 20 else None,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "trinucleotide_model_size": len(self.trinucleotide_probs)
        }
        
        # Add some stats about the trinucleotide model if it exists
        if self.trinucleotide_probs:
            # Top 5 most common trinucleotides
            top_trinucleotides = sorted(self.trinucleotide_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["top_trinucleotides"] = top_trinucleotides
        
        return stats
    
    def save_model(self, path):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_sequence': self.best_sequence,
            'historical_losses': self.historical_losses,
            'historical_sequences': self.historical_sequences,
            'trinucleotide_probs': self.trinucleotide_probs
        }, path)
    
    def load_model(self, path):
        """Load a saved model from disk"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.best_sequence = checkpoint['best_sequence']
            self.historical_losses = checkpoint['historical_losses']
            
            # Load trinucleotide model if it exists
            if 'historical_sequences' in checkpoint:
                self.historical_sequences = checkpoint['historical_sequences']
            else:
                self.historical_sequences = []
                
            if 'trinucleotide_probs' in checkpoint:
                self.trinucleotide_probs = checkpoint['trinucleotide_probs']
            else:
                # Build trinucleotide model from historical sequences
                self._build_trinucleotide_model()
                
            print(f"Model successfully loaded from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")