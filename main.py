from Bio import Entrez, SeqIO
from simulation.simulation import Simulation
from model.model import DNASequenceGenerator
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from tqdm import tqdm
import time



Entrez.email = "zpestrik@caltech.edu"

# Search for viral genomes in the nucleotide database
search_term = "Viruses[Organism]"
handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=1000)
record = Entrez.read(handle)
handle.close()

# Fetch the sequences
id_list = record["IdList"]
handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="gb", retmode="text")
sequences = list(SeqIO.parse(handle, "gb"))
handle.close()
model = DNASequenceGenerator()

print('Aquired Sequences, Initialized Model')

# Process the sequences as needed
best_loss = float('inf')
best_seq = None
i = 0
with open ('out.txt', 'w') as f:
    for seq_record in sequences:
        print(f'Sequence {i}')
        i += 1
        f.write(f"ID: {seq_record.id}\n")
        f.write(f"Description: {seq_record.description}\n")
        f.write(f"Sequence Length: {len(seq_record.seq)}\n")
        sim = Simulation(seq_record, 10, 100, 10000, 100)
        winner = sim.run(1000)
        f.write(f"Winner: {winner}\n")
        loss = sim.loss()
        f.write(f"Loss: {loss}\n")
        f.write('\n\n')
        model.train(seq_record.seq, loss)
        if loss < best_loss:
            best_loss = loss
            best_seq = seq_record

print('Trained Model')
print(f'Least Loss: {best_loss}')
print(f'Best Sequence: {best_seq.id}')
# Generate a sequence with the model
generated_seq = model.generate()
with open ('generated_sequence.txt', 'w') as f:
    f.write(f'GENERATED SEQUENCE with ESTIMATED LOSS {model.estimate_loss(generated_seq)}:\n\n{generated_seq[0]}')






