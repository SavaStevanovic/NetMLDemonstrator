class Identifier(object):
    def __init__(self):
        pass

    def get_identifier(self):
        idt = self.__class__.__name__
        if hasattr(self, 'inplanes'): 
            idt+='/'+str(self.inplanes)
        if hasattr(self, 'block_counts'): 
            idt+='/'+'-'.join([str(x) for x in self.block_counts])
        if hasattr(self, 'ratios'): 
            idt+='/'+'-'.join([str(x).replace('.',',') for x in self.ratios])
        if hasattr(self, 'block') and hasattr(self.block, 'get_identifier'): 
            idt+='/'+self.block.__name__
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'get_identifier'): 
            idt+='/'+self.backbone.get_identifier() 
        return idt
