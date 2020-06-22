import { Component, OnInit } from '@angular/core';
import { Filter } from '../filter';
import { FILTERS } from '../mock-filters';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.css']
})
export class FiltersComponent implements OnInit {

  filters = FILTERS;

  constructor() { }

  ngOnInit(): void {
  }

  onSelect(filter: Filter): void {
    filter.selected = !filter.selected;
  }
}
