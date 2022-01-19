import { Component, Input, EventEmitter, Output } from '@angular/core';
import { Filter } from 'src/app/models/filter';

@Component({
  selector: 'app-filter',
  templateUrl: './filter.component.html',
  styleUrls: ['./filter.component.css']
})
export class FilterComponent{
  @Input() filter: Filter;

  @Output() modelSelected = new EventEmitter();
  selectionChanged(): void {
    this.modelSelected.emit(this.filter.name);
  }
}
