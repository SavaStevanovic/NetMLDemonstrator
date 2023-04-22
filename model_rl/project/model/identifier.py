class Identifier:
    def __init__(self, block_counts, inplanes) -> None:
        self._block_counts = block_counts
        self._inplanes = inplanes
        
    def get_identifier(self):
        idt = "."
        idt += '/'+self.__class__.__name__
        idt += '/'+str(self._inplanes)
        idt += '/'+'-'.join([str(x) for x in self._block_counts])
        return idt