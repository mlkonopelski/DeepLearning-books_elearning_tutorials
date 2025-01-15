from typing import Optional
from datetime import date
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderLine:
    orderid: str
    sku: str
    qty: int


class Batch:
    def __init__(self, ref: str, sku: str, qty: int, eta: Optional[date]) -> None:
        self.ref = ref
        self.sku = sku
        self.eta = eta
        self._purchased_quantity = qty
        self._allocations: set[OrderLine] = set()

    def allocate(self, line: OrderLine) -> None:
        if self.can_allocate:
            self._allocations.add(line)
        
    def deallocate(self, line: OrderLine) -> None:
        if line in self._allocations:
            self._allocations.remove(line)

    @property
    def allocated_quantity(self) -> int:
        return sum(line.qty for line in self._allocations)

    @property
    def _available_quantity(self) -> int:
        return self._purchased_quantity - self.allocated_quantity

    def can_allocate(self, line: OrderLine) -> bool:
        return line.qty >= self._available_quantity and self.sku == line.sku



if __name__ == '__main__':

    ref = 'REF'
    sku = 'sku1'
    batch = Batch(ref, sku, 100, eta=date.today())
    line = OrderLine('order1', 'sku1', 10)

    batch.allocate(line)

    print(f'allocations: {batch._available_quantity}')
    print(batch._allocations)