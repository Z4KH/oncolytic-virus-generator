from Bio import SeqIO
import random

class Virus:

    num_viruses = 0

    def __init__(self, genome_record, num_viruses):
        self.genome_record = genome_record
        self.annotations = self.parse_genome(self.genome_record)
        self.probabilities = self.compute_probabilities(self.annotations)
        self.name = self.genome_record.name
        # Initialize virus count in constructor
        self.num_viruses = num_viruses  # Starting virus population

    def parse_genome(self, record):
        """
        Parse a GenBank file of a viral genome and analyze features related to:
        - Immune evasion
        - Replication efficiency
        - Natural tumor tropism factors
        - Cell entry mechanisms
        """
        annotations = {
            "immune_evasion": 0,
            "replication": 0,
            "tumor_tropism": 0,
            "cell_entry": 0,
            "genome_size": len(record.seq),
            "gc_content": 0,
            "ifn_antagonists": 0,
            "pkr_inhibitors": 0,
            "apoptosis_modulators": 0,
            "viral_family": self._determine_viral_family(record)
        }
        
        # Calculate GC content - influences stability and replication rate
        gc_count = record.seq.count('G') + record.seq.count('C')
        annotations["gc_content"] = (gc_count / len(record.seq)) * 100
        
        # Define natural viral features that contribute to tumor tropism
        immune_evasion_genes = {
            "ICP34.5", "ICP47", "US11", "US3", "US10", "V", "W", "C", "E3L", "K3L", 
            "NS1", "VP35", "NSP1", "ORF6", "Nef", "Vpu", "Vif"
        }
        
        replication_genes = {
            "NP", "P", "L", "F", "HN", "DNA polymerase", "helicase", "RdRp",
            "UL30", "UL42", "UL29", "E1", "E2", "RT", "Pol"
        }
        
        # Natural factors that contribute to tumor tropism without engineering
        natural_tumor_tropism_factors = {
            "thymidine kinase", "ribonucleotide reductase", # Reliance indicates tropism for dividing cells
            "p53 binding", "Rb binding", # Interaction with tumor suppressors
            "EGFR interaction", "PDGFR interaction", # Interactions with growth factor receptors
            "PKR inhibitor", "eIF2α", # Translation control factors exploited in cancer cells
            "Ras activation", "Akt signaling", # Oncogenic pathway interactions
            "IFN antagonist", "STAT inhibitor" # Interferon pathway interactions
        }
        
        cell_entry_genes = {
            "gB", "gC", "gD", "F", "H", "G", "spike", "S", "HA", "NA", "Env",
            "gp120", "gp41", "fiber", "penton"
        }
        
        # Scan gene product annotations and qualifiers
        for feature in record.features:
            if feature.type in ["CDS", "gene", "mat_peptide"]:
                for qualifier in ["gene", "product", "note"]:
                    if qualifier in feature.qualifiers:
                        for value in feature.qualifiers[qualifier]:
                            # Check for immune evasion genes
                            for gene in immune_evasion_genes:
                                if gene.lower() in value.lower():
                                    annotations["immune_evasion"] += 1
                                    break
                                    
                            # Check for interferon antagonists specifically
                            if any(term in value.lower() for term in ["interferon", "ifn", "stat", "jak"]):
                                annotations["ifn_antagonists"] += 1
                                
                            # Check for PKR inhibitors
                            if any(term in value.lower() for term in ["pkr", "eif2", "translation"]):
                                annotations["pkr_inhibitors"] += 1
                                
                            # Check for apoptosis modulators
                            if any(term in value.lower() for term in ["apoptosis", "bcl", "caspase", "death"]):
                                annotations["apoptosis_modulators"] += 1
                                
                            # Check for replication genes
                            for gene in replication_genes:
                                if gene.lower() in value.lower():
                                    annotations["replication"] += 1
                                    break
                                    
                            # Check for natural tumor tropism factors
                            for factor in natural_tumor_tropism_factors:
                                if factor.lower() in value.lower():
                                    annotations["tumor_tropism"] += 1
                                    break
                                    
                            # Check for cell entry genes
                            for gene in cell_entry_genes:
                                if gene.lower() in value.lower():
                                    annotations["cell_entry"] += 1
                                    break

        # Apply viral family-specific natural tropism modifiers
        family_tropism_modifier = {
            "herpesviridae": 1.2,  # HSV naturally targets dividing cells somewhat
            "adenoviridae": 1.5,   # Adenoviruses have some natural cancer cell preference
            "reoviridae": 2.0,     # Reoviruses have strong natural oncolytic activity
            "poxviridae": 1.1,     # Some natural tropism
            "paramyxoviridae": 1.8, # NDV has strong natural oncolytic effects
            "rhabdoviridae": 1.7,   # VSV exploits IFN defects
            "other": 1.0           # Default - no natural tropism
        }
        
        # Apply family-specific modifier to tumor tropism score
        family = annotations["viral_family"].lower()
        modifier = family_tropism_modifier.get(family, family_tropism_modifier["other"])
        annotations["tumor_tropism"] = annotations["tumor_tropism"] * modifier
        
        # Account for genome size in scoring (normalize by genome size)
        genome_size_kb = annotations["genome_size"] / 1000
        
        # Normalize scores by genome size
        for key in ["immune_evasion", "replication", "tumor_tropism", "cell_entry"]:
            annotations[key] = annotations[key] / genome_size_kb * 10
            
        return annotations

    def _determine_viral_family(self, record):
        """Determine viral family from GenBank record"""
        # Try to extract from source features
        for feature in record.features:
            if feature.type == "source":
                if "organism" in feature.qualifiers:
                    organism = feature.qualifiers["organism"][0].lower()
                    if "herpes" in organism:
                        return "herpesviridae"
                    elif "adeno" in organism:
                        return "adenoviridae"
                    elif "pox" in organism or "vaccinia" in organism:
                        return "poxviridae"
                    elif "reo" in organism:
                        return "reoviridae"
                    elif "newcastle" in organism or "paramyxo" in organism:
                        return "paramyxoviridae"
                    elif "vesicular" in organism or "rhabdo" in organism:
                        return "rhabdoviridae"
                    
        # Try to determine from description
        if hasattr(record, "description"):
            desc = record.description.lower()
            if "herpes" in desc:
                return "herpesviridae"
            elif "adeno" in desc:
                return "adenoviridae"
            elif "pox" in desc or "vaccinia" in desc:
                return "poxviridae"
            elif "reo" in desc:
                return "reoviridae"
            elif "newcastle" in desc or "paramyxo" in desc:
                return "paramyxoviridae"
            elif "vesicular" in desc or "rhabdo" in desc:
                return "rhabdoviridae"
                
        # Default if can't determine
        return "other"

    def compute_probabilities(self, ann):
        """
        Compute biologically relevant probabilities based on viral features.
        
        For natural tumor tropism, we consider:
        1. Interferon defects in cancer cells
        2. Cell cycle deregulation in cancer cells
        3. Metabolic differences
        4. Apoptosis resistance in cancer
        """
        # Basic immune evasion capability
        p_beat_immune = min(0.8, 0.01 + (0.01 * ann["immune_evasion"]) + random.uniform(-0.2, 0.2))
        
        # Natural tropism for cancer vs healthy cells
        # Cancer cells have defective antiviral responses
        ifn_defect_exploitation = min(0.5, 0.05 * ann["ifn_antagonists"]) 
        
        # Cancer cells have altered cell cycles
        cell_cycle_exploitation = 0.1 if ann["viral_family"] in ["herpesviridae", "adenoviridae"] else 0.05
        
        # Cancer cells have altered metabolism
        metabolism_exploitation = 0.02 * ann["gc_content"] / 50  # Normalized to typical GC content
        
        # Cancer cells often have defective apoptosis
        apoptosis_exploitation = min(0.3, 0.05 * ann["apoptosis_modulators"])
        
        # Reduced attack on healthy cells based on natural targeting mechanisms
        p_attack_healthy = max(0.1, 0.5 - (0.05 * ann["tumor_tropism"]) + random.uniform(-0.2, 0.2))
        
        # Enhanced attack on cancer cells - sum of exploitation factors
        p_attack_cancer = min(0.9, 0.3 + 
                             ifn_defect_exploitation + 
                             cell_cycle_exploitation + 
                             metabolism_exploitation + 
                             apoptosis_exploitation + random.uniform(-0.2, 0.2))
        
        # Replication efficiency
        p_reproduce = min(0.8, 0.001 + (0.01 * ann["replication"]) + random.uniform(-0.2, 0.2))
        
        # Use genome size as inverse indicator of replication speed
        # Smaller genomes generally replicate faster
        genome_size_factor = max(0.5, 1.5 - (ann["genome_size"] / 200000))  # Normalized to genome size ~200kb
        p_reproduce *= genome_size_factor
        
        return {
            "p_beat_immune": p_beat_immune,
            "p_attack_healthy": p_attack_healthy,
            "p_attack_cancer": p_attack_cancer,
            "p_reproduce": p_reproduce,
            "name": ann["viral_family"],
            "gc_content": ann["gc_content"],
            "genome_size": ann["genome_size"],
            "natural_tropism_score": ann["tumor_tropism"]
        }
    
    def step(self, num_immune_cells, num_healthy_cells, num_cancer_cells):
        """
        Simulate one step of virus interaction with host cells,
        modeling natural oncolytic activity with proper randomness.
        """
        # Constants based on viral kinetics literature
        BURST_SIZE = 50  # Average number of virions released per lysed cell
        IMMUNE_EFFICIENCY = 0.001  # Rate at which immune cells clear viruses
        
        # Calculate cell state differences between healthy and cancer cells
        CANCER_IFN_DEFECT = 0.7  # Cancer cells have ~70% reduced interferon response
        CANCER_APOPTOSIS_DEFECT = 0.6  # Cancer cells have ~60% reduced apoptosis
        CANCER_CELL_CYCLE = 2.0  # Cancer cells divide ~2x faster
        
        # Calculate adjusted probabilities based on cell states
        # For cancer cells - higher susceptibility due to defects
        cancer_infection_prob = self.probabilities["p_attack_cancer"]
        
        # For healthy cells - normal antiviral defense
        healthy_infection_prob = self.probabilities["p_attack_healthy"]
        
        # Apply interferon defect in cancer - increases infection probability
        cancer_infection_prob *= (1 + CANCER_IFN_DEFECT)
        
        # Apply cell cycle differences - viruses that depend on cell division
        if self.probabilities["name"] in ["herpesviridae", "adenoviridae", "poxviridae"]:
            cancer_infection_prob *= (1 + (CANCER_CELL_CYCLE - 1) * 0.5)
        
        # Initialize counters for infected cells
        healthy_cells_infected = 0
        cancer_cells_infected = 0
        immune_cells_affected = 0
        
        # Apply randomness at the individual virus level
        # Number of interaction attempts is proportional to virus population
        interaction_attempts = min(self.num_viruses * 10, 5000)  # Cap for performance
        
        # Probability distribution for what each virus particle encounters
        total_cells = num_healthy_cells + num_cancer_cells + num_immune_cells
        if total_cells > 0:
            p_encounter_healthy = num_healthy_cells / total_cells
            p_encounter_cancer = num_cancer_cells / total_cells
            p_encounter_immune = num_immune_cells / total_cells
        else:
            p_encounter_healthy = p_encounter_cancer = p_encounter_immune = 0
        
        # Track viruses used in infections (can't infect more than once)
        viruses_used = 0
        
        # Simulate individual virus particle interactions
        for _ in range(interaction_attempts):
            if viruses_used >= self.num_viruses:
                break  # Stop if all viruses are used
                
            # Determine which cell type is encountered
            rand_encounter = random.random()
            
            # Virus encounters healthy cell
            if rand_encounter < p_encounter_healthy:
                if random.random() < healthy_infection_prob:
                    if healthy_cells_infected < num_healthy_cells:
                        healthy_cells_infected += 1
                        viruses_used += 1
                        
            # Virus encounters cancer cell
            elif rand_encounter < p_encounter_healthy + p_encounter_cancer:
                if random.random() < cancer_infection_prob:
                    if cancer_cells_infected < num_cancer_cells:
                        cancer_cells_infected += 1
                        viruses_used += 1
                        
            # Virus encounters immune cell
            else:
                immune_evasion_factor = self.probabilities["p_beat_immune"]
                if random.random() < immune_evasion_factor:
                    # Virus evades immune cell
                    immune_cells_affected += 1
                else:
                    # Virus is cleared by immune cell
                    viruses_used += 1
        
        # Calculate additional immune clearance of free viruses
        remaining_viruses = self.num_viruses - viruses_used
        immune_clearance_rate = IMMUNE_EFFICIENCY * num_immune_cells
        viruses_cleared = int(random.uniform(0, immune_clearance_rate * remaining_viruses))
        viruses_cleared = min(viruses_cleared, remaining_viruses)
        
        # Viral replication - cancer cells produce more virus due to apoptosis defect
        new_viruses_from_cancer = 0
        if cancer_cells_infected > 0:
            # Add randomness to burst size
            for _ in range(cancer_cells_infected):
                # Random burst size with cancer cell enhancement
                cancer_burst_modifier = 1 + CANCER_APOPTOSIS_DEFECT
                burst = random.randint(BURST_SIZE // 2, BURST_SIZE * 2)
                
                # Apply replication efficiency with some randomness
                base_efficiency = self.probabilities["p_reproduce"]
                efficiency = random.uniform(base_efficiency * 0.7, base_efficiency * 1.3)
                
                new_viruses = int(burst * efficiency * cancer_burst_modifier)
                new_viruses_from_cancer += new_viruses
        
        # Healthy cells produce less virus
        new_viruses_from_healthy = 0
        if healthy_cells_infected > 0:
            for _ in range(healthy_cells_infected):
                # Random burst size for healthy cells (lower than cancer)
                burst = random.randint(BURST_SIZE // 3, BURST_SIZE)
                
                # Apply replication efficiency with some randomness
                base_efficiency = self.probabilities["p_reproduce"] * 0.7  # Reduced in healthy cells
                efficiency = random.uniform(base_efficiency * 0.7, base_efficiency * 1.3)
                
                new_viruses = int(burst * efficiency)
                new_viruses_from_healthy += new_viruses
        
        # Update virus population
        new_viruses = new_viruses_from_cancer + new_viruses_from_healthy
        total_used = viruses_used + viruses_cleared
        self.num_viruses = max(0, self.num_viruses - total_used + new_viruses)
        
        # Natural decay of virus particles
        decay_rate = random.uniform(0.05, 0.15)  # 5-15% decay per time step
        natural_decay = int(self.num_viruses * decay_rate)
        self.num_viruses -= natural_decay
        
        # Ensure virus count doesn't go negative
        self.num_viruses = max(0, self.num_viruses)
        
        return healthy_cells_infected, cancer_cells_infected, immune_cells_affected


# if __name__ == '__main__':
#     virus = Virus("data/sequence.gb", 10)
#     print(virus.probabilities)
#     for i in range(10):
#         print(virus.step(100, 1000, 1000))
#         print(virus.num_viruses)
#         print(f'\n\n')
