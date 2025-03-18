from simulation.virus import Virus

class Simulation:


    def __init__(self, virus, num_viruses, num_immune_cells, num_healthy_cells, num_cancer_cells):
        """
        Initialize a simulation with the given virus and cell counts.
        """
        self.virus = Virus(virus, num_viruses)
        self.num_immune_cells = num_immune_cells
        self.num_healthy_cells = num_healthy_cells
        self.num_cancer_cells = num_cancer_cells
        self.steps = 0
        self.winner = None

    def run(self, max_steps):
        """
        Run the simulation for a maximum number of steps.
        """
        while self.winner == None and self.steps < max_steps:
            self.winner = self.step()
            self.steps += 1
        
        if self.winner == None:
            if self.num_cancer_cells > self.num_healthy_cells:
                self.winner = "cancer"
            else:
                self.winner = "healthy"
        return self.winner

    def step(self):
        """
        Perform one step of the simulation.
        """
        healthy_cells_dead, cancer_cells_dead, immune_cells_dead = self.virus.step(self.num_immune_cells, self.num_healthy_cells, self.num_cancer_cells)
        self.num_healthy_cells -= healthy_cells_dead
        self.num_cancer_cells -= cancer_cells_dead
        self.num_immune_cells -= immune_cells_dead
        if self.num_healthy_cells == 0:
            return "cancer"
        elif self.num_cancer_cells == 0 and self.num_healthy_cells > Virus.num_viruses:
            return "healthy"
        return None
    
    def loss(self):
        """
        Compute the loss function of the simulation.
        """
        loss = 0
        if self.winner == "cancer":
            loss += 10000
            loss -= self.steps * 10
            loss = max(0, loss)
        elif self.winner == "healthy":
            loss -= 100
            loss += self.steps * 10
            loss = max(0, loss)
        loss += self.virus.num_viruses / 50
        loss -= self.num_immune_cells
        return max(0, loss)