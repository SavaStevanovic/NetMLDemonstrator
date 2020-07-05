import { Component, OnInit, Input } from '@angular/core';
import { Filter } from 'src/app/models/filter';

@Component({
  selector: 'app-filter',
  templateUrl: './filter.component.html',
  styleUrls: ['./filter.component.css']
})
export class FilterComponent implements OnInit {
  @Input() filter: Filter;

  constructor() { }

  ngOnInit(): void {
  }

  onSelect(filter: Filter): void {
    filter.selected = !filter.selected;
  }
}
