class EarlyStopping:
    def __init__(self, patience: int, mode: str = 'min'):
        # TODO: store patience, mode, best (initialized to +/-inf), and a bad-step counter
        self.patience = patience
        self.patience_counter=patience
        self.mode = mode
        if(mode=='min'):
            self.best=float('+inf')
        else:
            self.best=float('-inf')

    def step(self, metric: float) -> bool:
        # TODO: update best/counter, return True iff should stop
        if(self.mode=='min'):
            if(metric<self.best):
                self.best=metric
                self.patience_counter=self.patience
            else:
                self.patience_counter-=1
            if self.patience_counter<=0:
                return True
            return False
        if(metric>self.best):
            self.best=metric
            self.patience_counter=self.patience
        else:
            self.patience_counter-=1
        if self.patience_counter<=0:
            return True
        return False