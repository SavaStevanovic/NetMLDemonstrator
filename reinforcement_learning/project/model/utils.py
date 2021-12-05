import abc


class Identifier(abc.ABC):
    def __init__(self):
        pass

    def get_identifier(self):
        idt = self.__class__.__name__
        idt += '/'+str(self.inplanes)
        idt += '/'+'-'.join([str(x) for x in self.block_counts])
        if hasattr(self, '_block'):
            idt += '/'+self._block.__name__
        if hasattr(self, '_backbone') and isinstance(self._backbone, Identifier):
            idt += '/'+self._backbone.get_identifier()
        return idt

    @property
    @abc.abstractmethod
    def inplanes(self):
        pass

    @property
    @abc.abstractmethod
    def block_counts(self):
        pass
